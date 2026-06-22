#Calculates approximate distance per pixel. 
# Input: DEM, camera position 
# Output: Murtel_distance.csv; Murtel_distance_filled.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import Delaunay
from scipy.ndimage import distance_transform_edt
import trimesh
from pyproj import Transformer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
IMAGE_WIDTH  = 336
IMAGE_HEIGHT = 252
FOV_DEG      = 35
BATCH_SIZE   = 10

CAM_LAT, CAM_LON, CAM_Z = 46.431084, 9.822692, 2697.8
TGT_LAT, TGT_LON, TGT_Z = 46.429452, 9.821723, 2655.9

XYZ_FILE     = os.path.join(DATA_DIR, "DEM", "SWISSALTI3D_0.5_XYZ_CHLV95_LN02_2783_1144.xyz")
OUT_DIR      = os.path.join(DATA_DIR, "CSVs")
SITE_NAME    = "Murtel"

TIR_IMG_PATH = "insert background TIR image path"

# Fill values below this distance threshold as invalid (meters)
MIN_VALID_DIST = 100

# Overlay alpha (0 = TIR only, 1 = depth map only)
OVERLAY_ALPHA = 0.8


def to_lv95(lat, lon, z):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.array([x, y, z])


def load_mesh(xyz_file):
    xyz = np.loadtxt(xyz_file, skiprows=1)
    points_3d = xyz[:, :3]
    tri = Delaunay(points_3d[:, :2])
    return trimesh.Trimesh(vertices=points_3d, faces=tri.simplices, process=False)


def generate_camera_rays(fov_rad, width, height):
    cx, cy = width / 2, height / 2
    fx = width / (2 * np.tan(fov_rad / 2))
    rays = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            ray = np.array([(j - cx) / fx, (i - cy) / fx, 1.0])
            rays[i, j] = ray / np.linalg.norm(ray)
    return rays


def get_rotation_matrix(camera_pos, target_pos):
    forward = target_pos - camera_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, -1])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    true_up /= np.linalg.norm(true_up)
    return np.stack((right, true_up, forward), axis=1)


def cast_rays(mesh, camera_pos, rays_cam, batch_size):
    R = get_rotation_matrix(camera_pos, to_lv95(TGT_LAT, TGT_LON, TGT_Z))
    rays_flat = (R @ rays_cam.reshape(-1, 3).T).T.astype(np.float32)
    n = len(rays_flat)
    origins = np.tile(camera_pos, (n, 1)).astype(np.float32)
    distances = np.full(n, np.nan, dtype=np.float32)

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        print(f"Batch {end}/{n}")
        locs, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins[i:end],
            ray_directions=rays_flat[i:end]
        )
        if len(idx_ray):
            distances[i + idx_ray] = np.linalg.norm(locs - origins[i + idx_ray], axis=1)

    return distances


def fill_nearest(arr, mask):
    filled = arr.copy()
    _, inds = distance_transform_edt(mask, return_indices=True)
    filled[mask] = arr[tuple(inds[:, mask])]
    return filled


def main():
    camera_pos = to_lv95(CAM_LAT, CAM_LON, CAM_Z)
    fov_rad    = np.radians(FOV_DEG)

    print("Loading mesh...")
    mesh = load_mesh(XYZ_FILE)

    print("Generating rays...")
    rays_cam = generate_camera_rays(fov_rad, IMAGE_WIDTH, IMAGE_HEIGHT)

    print("Casting rays...")
    distances = cast_rays(mesh, camera_pos, rays_cam, BATCH_SIZE)

    # Save raw distances
    raw_csv = os.path.join(OUT_DIR, f"{SITE_NAME}_distance.csv")
    pd.DataFrame({"distances": distances}).to_csv(raw_csv, index=False)
    print(f"Saved raw distances: {raw_csv}")

    # Fill invalid distances
    depth = distances.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    mask_bad = (~np.isfinite(depth)) | (depth < MIN_VALID_DIST)
    depth_filled = fill_nearest(depth, mask_bad)

    filled_csv = os.path.join(OUT_DIR, f"{SITE_NAME}_distance_filled.csv")
    pd.DataFrame({"distances_filled": depth_filled.flatten()}).to_csv(filled_csv, index=False)
    print(f"Saved filled distances: {filled_csv}")

    # Load TIR image for overlay
    tir_img = cv2.imread(TIR_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if tir_img is None:
        raise FileNotFoundError(f"Could not read TIR image: {TIR_IMG_PATH}")
    tir_img   = cv2.resize(tir_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    tir_rgb   = cv2.cvtColor(tir_img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    # Depth colormap and overlay
    vmax      = np.nanmax(depth_filled)
    norm_cb   = plt.Normalize(vmin=0.0, vmax=vmax)
    depth_norm = norm_cb(depth_filled)
    depth_rgb  = plt.cm.viridis(np.power(depth_norm, 0.5))[..., :3]
    overlay    = np.clip((1 - OVERLAY_ALPHA) * tir_rgb + OVERLAY_ALPHA * depth_rgb, 0, 1)

    # Plot
    ticks = np.arange(0, 501, 100)
    fig = plt.figure(figsize=(15, 4))
    gs  = fig.add_gridspec(1, 3, wspace=0.2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    im0 = ax0.imshow(depth,        cmap="viridis", norm=norm_cb)
    im1 = ax1.imshow(depth_filled, cmap="viridis", norm=norm_cb)
    ax2.imshow(overlay)

    ax0.set_title("Raw distance map")
    ax1.set_title("Gap-filled distance map")
    ax2.set_title("Distance map overlaid on TIR image")

    for ax in (ax0, ax1, ax2):
        ax.axis("off")

    def add_cbar(fig_, ax_, im_, width=0.012, gap=0.005, ticks_=None):
        pos = ax_.get_position()
        cax = fig_.add_axes([pos.x1 + gap, pos.y0, width, pos.height])
        cb  = fig_.colorbar(im_, cax=cax, ticks=ticks_)
        cb.ax.tick_params(pad=4, labelsize=10)

    add_cbar(fig, ax0, im0, ticks_=ticks)
    add_cbar(fig, ax1, im1, ticks_=ticks)

    plt.show()


if __name__ == "__main__":
    main()
