
<p align="center">
  <img width="416" alt="09fc432b820c711e9b2aa1a988076e0" src="https://github.com/user-attachments/assets/11f8a970-fae2-4fa3-b553-cce416af309f" />
</p>

<h1 align="center"> GEOL0069_Echo_Classification</h1>
<p align="center">Classify the echoes in leads and sea ice and produce an average echo shape as well as standard deviation for these two classes.
 </p>
<br />
<!-- This is my very first GitHub page -->
<!-- This is really fun to play with -->
<!-- If you saw my page then thank you so much for stopping by!!! -->
<!-- Since it is my first GitHub page, if there are anything that needs to be improved, please do tell me-->
<!-- Let's go and finish this :))))))))))))))) -->

# About The Project

This project focuses on colocating Sentinel-3 (OLCI & SRAL) and Sentinel-2 optical data to enhance Earth Observation (EO) analysis. By integrating these datasets, we leverage Sentinel-2's high spatial resolution alongside Sentinel-3's broad coverage and altimetry data, creating a richer perspective of Earth's surface.

Using unsupervised learning, the project aims to classify sea ice and leads, enabling automated environmental feature detection. The ultimate goal is to develop a scalable pipeline that fuses multi-sensor satellite data and applies machine learning models to improve sea ice classification and EO applications.

<details>
  <summary>Table of Contents</summary>
  <ol>
  </ol>
</details>

## Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data
In this section, we explore the integration of Sentinel-3 and Sentinel-2 optical data, leveraging their combined strengths. By aligning data from these two satellite missions, we harness Sentinel-2’s high spatial resolution alongside Sentinel-3’s extensive coverage and colocated altimetry, creating a more comprehensive and detailed view of Earth’s surface.

### Step 0: Read in Functions Needed
Before we start our work, first we perform a data load, we will mount Colab and Google Drive, subsequently we need to retrieve the metadata of Sentinel 2 and Sentinel 3. Subsequently, by using the code, entering the account password, scripting to get the access token, and logging into The Copernicus Data Space Ecosystem to get the data. This is the main prerequisite for visualizing Sentinel 2 and 3 data.

### Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)
In this step we will co-locate Sentinel II and Sentinel III OLCIs by retrieving their metadata. We name them 'sentinel3_olci_data' and 'sentinel2_data' to categorize and retrieve the data respectively
