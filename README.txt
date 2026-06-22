Surface Temperature Dynamics at the Murtèl Rock Glacier

Code accompanying the master's thesis Spatio-temporal surface temperature dynamics at the Murtèl rock glacier
by Marco Hassler, University of Zurich, 2026.

This repository contains the processing and Analysis Pipelines used to derive and interpret surface temperatures from thermal infrared (TIR) imagery of the Murtèl rock glacier (Murtèl-Corvatsch permafrost site, Swiss Alps, ~2600 m a.s.l.). The workflow combines (1) preprocessing of the used data, (2) atmospheric correction of Mobotix TIR images, and (3) statistical analysis (k-means clustering, z-normalisation, multiple linear regression, GAMs) of how meteorological drivers control surface temperature patterns.

----------------------------------------------------------

Repository structure:


codes/
├── preprocessing/
│   └── meteo_filtering.py        	# Filter and clean Corvatsch meteorological data							 						# Figure 6, Figure A.1-A.3
│   └── BH_GST_filtering.py     	# Filter and clean borehole and GST data (in this study, only used for plotting purposes)		 	# Figure 7
│   └── decode_TIR_to_CSV.py      	# Convert TIR PNGs to CSVs
│   └── label_TIR_images.py       	# Labelling Tool for TIR images to enable entropy calculation
│   └── calc_filter_entropy.py  	# Calculates entropy based on labelled images and filters raw CSVs based on this entropy		 	# Figure 5
│   └── distance_calculation.py 	# Calculates approximate distance per pixel based on DEM and camera popsition				 		# Figure 8
│   └── extract_ICT.py    			# Extracts internal camera temperature from TIR PNGs
│
├── atmospheric_correction/
│   └── correction.py 					# Iterates through raw TIR CSV folder to apply temperature correction.
│   └── extract_pixel_timeseries.py 	# Extracts temperatures at validation measurement point locations (met station, boreholes, GSTs, radiometers) from both the raw and atmospherically corrected CSVs. It then attaches the nearest
│					  					  meteorological and validation data (emissivity, air temperature, radiation, borehole temperatures, GST, radiometer readings) and saves everything as a single time series CSV
└── analysis/
│   └── temperature_curve_grouping.py						# Groups pixels in chosen analysis ranges based on the diurnal curve clustering algorithm
│   └── anomaly_peristence_grouping.py						# Groups pixels in chosen analysis ranges based on the anomaly persistence clustering algorithm
│   └── temperature_curve_regression.py					# Applies MLR to the previously defined groups based on the diurnal curve clustering algorithm						# Figure 21-28
│   └── anomaly_persistence_regression.py					# Applies MLR to the previously defined groups based on the anomaly persistence clustering algorithm.				# Figure 29-34
│   └── temperature_curve_regression_postprocessing.py		# Plots the standardized relative influence of each meteorological predictor per cluster over time based on the diurnal curve clustering algorithm
│   └── anomaly_persistence_regression_postprocessing.py	# Plots the standardized relative influence of each meteorological predictor per cluster over time based on the anomaly persistence clustering algorithm
│   └── temperature_curve_gam.py							# Applies GAM calculation to the previously defined days based on the diurnal curve clustering algorithm			# Figure C.17-C.41
│   └── anomaly_persistence_gam.py							# Applies GAM calculation to the previously defined days based on the anomaly persistence clustering algorithm	 	# Figure C.42-C.67
│
└── plotting/
    └── temperature_curve_clustering_visualization.py			# Figure 9
    └── validation_plots.py									# Figure 10-12, B.1-B.9
    └── dT_distance_plots.py									# Figure 13, B.10-B.13
    └── TIR_vs_validation_temperatures.py						# Figure 14-16, B.14-B.24
    └── aluminium_correction_comparison_plots.py				# Figure 17-19, B.47-B.49
    └── barplot_silhouette_scores.py							# Figure 20
    └── period_influence_boxplots.py							# Figure 35-36
    └── emissivity_transmissivity_scatter_comparison_plots.py	# Figure B.25-B.46
    └── relative_influence_plots.py							# Figure C.1-C.16


----------------------------------------------------------

Requirements
Python > 3.13

Libraries: csv, cv2, datetime, __future__, gc, glob, imageio.v2, math, matplotlib, numpy, os, pandas, pygam, pyproj, random, re, shutil, skearn, statsmodels, subprocess, trimesh, typing, warnings, zoneinfo

All scripts are designed to run directly in VS Code.

----------------------------------------------------------

Usage

Data:
CSVs/
	└── murtel_met.csv	(raw Murtèl meteorological station data)
validation_data/
	└── BH_1			(Borehole 1 file)
	└── BH_2			(Borehole 2 file)
	└── BH_3			(Borehole 3 file)
	└── GST				(file containg all ground surface temperature measurement data)
	└── meteo_piz_corvatsch 	(raw Corvatsch meteorological station data)
	└── radiometer_furrow	 	(radiometer furrow data)
	└── radiometer_ridge	 	(radiometer ridge data)
DEM/
	└── SWISSALTI3D_0.5_XYZ_CHLV95_LN02_2783_1144.xyz	(DEM file)
TIR_camera/
	└── TIR_raw_data	(TIR PNG images)
	└── RGB_images		(RGB PNG images)
glacier_mask.png 		(to mask out foreground and background)
TIR_mask.png			(to mask out foreground)


Preprocessing
meteo_filtering.py: Filters the raw Murtèl meteorological station data to the study period, removes unrealistic values and writes a cleaned CSV used by all downsteam analyses. Input: murtel_met.csv | Output: murtel_met_qc.csv; 	(optional: raw and filtered comparison plots of all important variables)
decode_TIR_to_CSV.py: Decodes output PNGs from the mobotix camera to CSVs with °C per pixel and corrects for daylight saving time. Input: MOBOTIX decoder path; Raw TIR PNGs |Output: TIR CSVs containing °C values per pixel
label_TIR_images.py: Tool that enables manual labelling of 1000 random TIR images. Used for further entropy calculation. Input: TIR and RGB images |Output: labels.csv
calc_filter_entropy.py: Calulates entropy threshold based on labels.csv and then copies all good images to a new subfolder. Input: labels.csv; raw TIR CSVs path |Output: New subfolder with only good images
distance_calculation.py: Calculates approximate distance per pixel. Input: DEM, camera position | Output: Murtel_distance.csv; Murtel_distance_filled.csv
extract_ICT.py: Extracts internal camera temperature from TIR PNGs. Input: TIR PNG folder | Output: internal_camera_temps.csv

2. Atmospheric Correction
correction.py: Iterates through a folder containing all TIR CSVs to apply the atmospheric correction on those. Input: raw TIR folder path; output path; Murtel_distance_filled.csv; murtel_met_qc.csv | Output: Corrected CSVs; 	daily_emissivity.csv
extract_pixel_timeseries.py:  Extracts temperatures at validation measurement point locations (met station, boreholes, GSTs, radiometers) from both the raw and atmospherically corrected CSVs. It then attaches the nearest meteorological 	and validation data (emissivity, air temperature, radiation, borehole temperatures, GST, radiometer readings) and saves everything as a single time series CSV. Input: validation datasets; raw and corrected TIR CSVs, 	murtel_met_qc.csv | Output: pixel_timeseries.csv

3. Analysis
temperature_curve_grouping.py: Applies KMeans clustering (k=2-7) to mean diurnal TIR temperature curves using a rolling 5-day window, one target day at a time. Groups pixels by their daily temperature curve shape across the chosen a	nalysis periods. Input: corrected CSV folder, glacier_mask.png, TIR_mask.png, pixel_timeseries.csv | Output: cluster label CSVs, cluster curve CSVs and PNGs, cluster map PNGs, silhouette summary CSV
anomaly_persistence_grouping.py:  Counts per-pixel how often each image's pixel value is above the image mean (z > 0) within each rolling 5-day window. Groups pixels into persistently cold, intermediate, and persistently warm classes 	based on quantiles of these counts. Input: corrected CSV folder, glacier_mask.png/TIR_mask.png pixel_timeseries.csv | Output: count_above CSVs, spatial maps (count above/below PNGs), per image z-score frames
temperature_curve_regression.py: Fits a rolling-window multiple linear regression per cluster (Cluster 1 / Cluster 2) predicting mean cluster temperature from meteorological predictors. Input: corrected CSV folder, 	temperature_curve_grouping outputs, pixel_timeseries.csv | Output: regression coefficient CSVs, cluster timeseries CSVs, quality report CSVs, GIFs of daily cluster masks, daily mean curve plots
anomaly_persistence_regression.py: Fits a rolling-window standardized multiple linear regression per pixel group (cold/mid/warm) predicting mean group temperature from meteorological predictors. Input: corrected CSV folder, 	anomaly_persistence_grouping outputs, pixel_timeseries.csv | Output: regression coefficient CSVs, group timeseries CSVs, quality report CSVs, GIFs of daily group masks
temperature_curve_regression_postprocessing.py: Reads the regression outputs and plots the standardized relative influence of each meteorological predictor per cluster over time, alongside R² time series. Input: 	temperature_curve_regression outputs | Output: relative influence PNGs, residual PNGs per cluster
anomaly_persistence_regression_postprocessing.py: Same as above but for the anomaly persistence pixel groups (cold/mid/warm). Input: anomaly_persistence_regression outputs | Output: relative influence PNGs, residual PNGs per group and 	combined
temperature_curve_gam.py: Fits a Generalized Additive Model (GAM) per cluster using the same meteorological predictors as the multiple linear regression and generates partial dependence plots (PDPs) showing the non-linear effect of each 	predictor. Input: temperature_curve_regression outputs (cluster timeseries CSVs) | Output: PDP PNGs per selected target day
anomaly_persistence_gam.py:  Same as above but for the anomaly persistence pixel groups (cold/mid/warm). Input: anomaly_persistence_regression outputs (group timeseries CSVs) | Output: PDP PNGs per selected target day