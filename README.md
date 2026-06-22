#### **Surface Temperature Dynamics at the Murtèl Rock Glacier**

Code accompanying the master's thesis "Spatio-Temporal Surface Temperature Dynamics and Atmospheric Controls on Murtèl Rock Glacier"

by Marco Hassler, University of Zurich, 2026.

This repository contains the processing and Analysis Pipelines used to derive and interpret surface temperatures from thermal infrared (TIR) imagery of the Murtèl rock glacier (Murtèl-Corvatsch permafrost site, Swiss Alps, \~2600 m a.s.l.). The workflow combines (1) preprocessing of the used data, (2) atmospheric correction of Mobotix TIR images, and (3) statistical analysis (k-means clustering, z-normalisation, multiple linear regression, GAMs) of how meteorological drivers control surface temperature patterns.

\----------------------------------------------------------

###### **Repository structure:**


**codes/**

**├── preprocessing/**

**│   └── meteo\_filtering.py        	# Filter and clean Corvatsch meteorological data							 # Figure 6, Figure A.1-A.3**

**│   └── BH\_GST\_filtering.py        	# Filter and clean borehole and GST data (in this study, only used for plotting purposes)		 # Figure 7**

**│   └── decode\_TIR\_to\_CSV.py      	# Convert TIR PNGs to CSVs**

**│   └── label\_TIR\_images.py       	# Labelling Tool for TIR images to enable entropy calculation**

**│   └── calc\_filter\_entropy.py     	# Calculates entropy based on labelled images and filters raw CSVs based on this entropy		 # Figure 5**

**│   └── distance\_calculation.py    	# Calculates approximate distance per pixel based on DEM and camera popsition				  # Figure 8**

**│   └── extract\_ICT.py    		# Extracts internal camera temperature from TIR PNGs**

**│**

**├── atmospheric\_correction/**

**│   └── correction.py 			# Iterates through raw TIR CSV folder to apply temperature correction.**

**│   └── extract\_pixel\_timeseries.py 	# Extracts temperatures at validation measurement point locations (met station, boreholes, GSTs, radiometers) from both the raw and atmospherically corrected CSVs. It then attaches the nearest**

**│					  meteorological and validation data (emissivity, air temperature, radiation, borehole temperatures, GST, radiometer readings) and saves everything as a single time series CSV**

**└── analysis/**

**│   └── temperature\_curve\_grouping.py				# Groups pixels in chosen analysis ranges based on the diurnal curve clustering algorithm**

**│   └── anomaly\_peristence\_grouping.py				# Groups pixels in chosen analysis ranges based on the anomaly persistence clustering algorithm**

**│   └── temperature\_curve\_regression.py				# Applies MLR to the previously defined groups based on the diurnal curve clustering algorithm				# Figure 21-28**

**│   └── anomaly\_persistence\_regression.py			# Applies MLR to the previously defined groups based on the anomaly persistence clustering algorithm.			# Figure 29-34**

**│   └── temperature\_curve\_regression\_postprocessing.py		# Plots the standardized relative influence of each meteorological predictor per cluster over time based on the diurnal curve clustering algorithm**

**│   └── anomaly\_persistence\_regression\_postprocessing.py	# Plots the standardized relative influence of each meteorological predictor per cluster over time based on the anomaly persistence clustering algorithm**

**│   └── temperature\_curve\_gam.py				# Applies GAM calculation to the previously defined days based on the diurnal curve clustering algorithm		# Figure C.17-C.41**

**│   └── anomaly\_persistence\_gam.py				# Applies GAM calculation to the previously defined days based on the anomaly persistence clustering algorithm	 	# Figure C.42-C.67**

**│**

**└── plotting/**

&#x20;   **└── temperature\_curve\_clustering\_visualization.py		# Figure 9**

&#x20;   **└── validation\_plots.py					# Figure 10-12, B.1-B.9**

&#x20;   **└── dT\_distance\_plots.py					# Figure 13, B.10-B.13**

&#x20;   **└── TIR\_vs\_validation\_temperatures.py			# Figure 14-16, B.14-B.24**

&#x20;   **└── aluminium\_correction\_comparison\_plots.py		# Figure 17-19, B.47-B.49**

&#x20;   **└── barplot\_silhouette\_scores.py				# Figure 20**

&#x20;   **└── period\_influence\_boxplots.py				# Figure 35-36**

&#x20;   **└── emissivity\_transmissivity\_scatter\_comparison\_plots.py	# Figure B.25-B.46**

&#x20;   **└── relative\_influence\_plots.py				# Figure C.1-C.16**

\----------------------------------------------------------

###### **Requirements**

* Python > 3.13

Libraries: csv, cv2, datetime, \_\_future\_\_, gc, glob, imageio.v2, math, matplotlib, numpy, os, pandas, pygam, pyproj, random, re, shutil, skearn, statsmodels, subprocess, trimesh, typing, warnings, zoneinfo

**All scripts are designed to run directly in VS Code.**

\----------------------------------------------------------

###### **Usage**

**Data:**

* **CSVs/**

&#x09;**└── murtel\_met.csv	(raw Murtèl meteorological station data)**

* **validation\_data/**

&#x09;**└── BH\_1			(Borehole 1 file)**

&#x09;**└── BH\_2			(Borehole 2 file)**

&#x09;**└── BH\_3			(Borehole 3 file)**

&#x09;**└── GST				(file containg all ground surface temperature measurement data)**

&#x09;**└── meteo\_piz\_corvatsch 	(raw Corvatsch meteorological station data)**

&#x09;**└── radiometer\_furrow	 	(radiometer furrow data)**

&#x09;**└── radiometer\_ridge	 	(radiometer ridge data)**

* **DEM/**

&#x09;**└── SWISSALTI3D\_0.5\_XYZ\_CHLV95\_LN02\_2783\_1144.xyz	(DEM file)**

* **TIR\_camera/**

&#x09;**└── TIR\_raw\_data	(TIR PNG images)**

&#x09;**└── RGB\_images		(RGB PNG images)**

* **glacier\_mask.png 		(to mask out foreground and background)**
* **TIR\_mask.png			(to mask out foreground)**


1. **Preprocessing**
* **meteo\_filtering.py:** Filters the raw Murtèl meteorological station data to the study period, removes unrealistic values and writes a cleaned CSV used by all downsteam analyses. **Input: murtel\_met.csv | Output: murtel\_met\_qc.csv;** (optional: raw and filtered comparison plots of all important variables)
* **decode\_TIR\_to\_CSV.py:** Decodes output PNGs from the mobotix camera to CSVs with °C per pixel and corrects for daylight saving time. **Input: MOBOTIX decoder path; Raw TIR PNGs |Output: TIR CSVs containing °C values per pixel**
* **label\_TIR\_images.py:** Tool that enables manual labelling of 1000 random TIR images. Used for further entropy calculation. **Input: TIR and RGB images |Output: labels.csv**
* **calc\_filter\_entropy.py:** Calulates entropy threshold based on labels.csv and then copies all good images to a new subfolder. **Input: labels.csv; raw TIR CSVs path |Output: New subfolder with only good images**
* **distance\_calculation.py:** Calculates approximate distance per pixel. **Input: DEM, camera position | Output: Murtel\_distance.csv; Murtel\_distance\_filled.csv**
* **extract\_ICT.py:** Extracts internal camera temperature from TIR PNGs. **Input: TIR PNG folder | Output: internal\_camera\_temps.csv**


2\. **Atmospheric Correction**
* **correction.py:** Iterates through a folder containing all TIR CSVs to apply the atmospheric correction on those. **Input: raw TIR folder path; output path; Murtel\_distance\_filled.csv; murtel\_met\_qc.csv | Output: Corrected CSVs; daily\_emissivity.csv**
* **extract\_pixel\_timeseries.py:**  Extracts temperatures at validation measurement point locations (met station, boreholes, GSTs, radiometers) from both the raw and atmospherically corrected CSVs. It then attaches the nearest meteorological and validation data (emissivity, air temperature, radiation, borehole temperatures, GST, radiometer readings) and saves everything as a single time series CSV. **Input: validation datasets; raw and corrected TIR CSVs, murtel\_met\_qc.csv | Output: pixel\_timeseries.csv**


3\. **Analysis**
* **temperature\_curve\_grouping.py:** Applies KMeans clustering (k=2) to mean diurnal TIR temperature curves using a rolling 5-day window, one target day at a time. Groups pixels by their daily temperature curve shape across the chosen analysis periods. **Input: corrected CSV folder, glacier\_mask.png, pixel\_timeseries.csv | Output: cluster label CSVs, cluster curve CSVs, cluster map PNGs, silhouette summary CSV**
* **anomaly\_persistence\_grouping.py:**  Counts per-pixel how often each image's pixel value is above the image mean (z > 0) within each rolling 5-day window. Groups pixels into persistently cold, intermediate, and persistently warm classes based on quantiles of these counts. **Input: corrected CSV folder, glacier\_mask.png, pixel\_timeseries.csv | Output: count\_above CSVs, spatial maps, pixel count tables, window summary** CSV
* **temperature\_curve\_regression.py:** Fits a rolling-window multiple linear regression per cluster (Cluster 1 / Cluster 2) predicting mean cluster temperature from meteorological predictors. **Input: corrected CSV folder, temperature\_curve\_grouping outputs, pixel\_timeseries.csv | Output: regression coefficient CSVs, cluster timeseries CSVs, quality report CSVs, GIFs of daily cluster masks, daily mean curve plots**
* **anomaly\_persistence\_regression.py:** Fits a rolling-window standardized multiple linear regression per pixel group (cold/mid/warm) predicting mean group temperature from meteorological predictors. **Input: corrected CSV folder, anomaly\_persistence\_grouping outputs, pixel\_timeseries.csv | Output: regression coefficient CSVs, group timeseries CSVs, quality report CSVs, GIFs of daily group masks**
* **temperature\_curve\_regression\_postprocessing.py:** Reads the regression outputs and plots the standardized relative influence of each meteorological predictor per cluster over time, alongside R² time series. **Input: temperature\_curve\_regression outputs | Output: relative influence PNGs, residual PNGs per cluster**
* **anomaly\_persistence\_regression\_postprocessing.py:** Same as above but for the anomaly persistence pixel groups (cold/mid/warm). **Input: anomaly\_persistence\_regression outputs | Output: relative influence PNGs, residual PNGs per group and combined**
* **temperature\_curve\_gam.py:** Fits a Generalized Additive Model (GAM) per cluster using the same meteorological predictors as the multiple linear regression and generates partial dependence plots (PDPs) showing the non-linear effect of each predictor. **Input: temperature\_curve\_regression outputs (cluster timeseries CSVs) | Output: PDP PNGs per selected target day**
* **anomaly\_persistence\_gam.py:**  Same as above but for the anomaly persistence pixel groups (cold/mid/warm). **Input: anomaly\_persistence\_regression outputs (group timeseries CSVs) | Output: PDP PNGs per selected target day**

