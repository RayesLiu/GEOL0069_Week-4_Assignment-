
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
In this step we will co-locate Sentinel III OLCIs and Sentinel II by retrieving their metadata. We name them 'sentinel3_olci_data' and 'sentinel2_data' to categorize and retrieve the data respectively. The following is the generated data, generated in csv format, so we can clearly see the Sentinel II and Sentinel III data ID, naming, ContentType, ContentLength, and data logging time and so on.

#### Sentinel III
<img width="1172" alt="fdf1105efd34658ffca4d0d3565c83f" src="https://github.com/user-attachments/assets/ac158bc4-b7d8-483d-bc79-5f01cfa49a79" />

#### Sentinel II
<img width="1184" alt="57c01a9d3ade32693594e9912160078" src="https://github.com/user-attachments/assets/50a0acf5-0806-4afa-b8f1-52936c0edd33" />
With the information from the above two tables, a large amount of data is retrieved, at which point we need to co-locate this metadata and generate a new table.

#### Co-locate the metadata
In this section we use the metadata we have just produced to produce the co-location pair details. The logic of the code is match rows from S2 and S3 OLCI by their geo_footprint.

![image](https://github.com/user-attachments/assets/4133af0d-090a-4f56-b47a-6e51ff3b503d)

This table generates data from five collocated satellites, which contains important information such as satellite name, ID, footprint and overlap time to determine the satellite co-location data and make further observations.

![image](https://github.com/user-attachments/assets/d7d5c3d9-5916-4922-8013-0af1a09751e8)

This image is a visualization of the collocation footprint of the Sentinel II and Sentinel III OLCIs. The blue area is the overlap observation between the two satellites.

#### Proceeding with Sentinel-3 OLCI Download
Next, we focus on downloading Sentinel-3 OLCI data, following a similar approach to Sentinel-2 to ensure methodological consistency. Using the same filename conversion logic, we adhere to a structured workflow to efficiently retrieve data from the Copernicus Data Space.

##### Sentinel-3 SRAL

Here we can also collocate Sentinel II's Sentinel III OLCI altimetry data. data retrieval is similar to the above steps, we will only need to obtain Sentinel III's SRAL metadata.
After acquiring the data, collocate it with Sentinel II. we can then get a new visualization of the graph.

![image](https://github.com/user-attachments/assets/1310937f-7f30-4f4a-aa41-36f6680ef23f)

## Unsupervised Learning

This section introduces the practical application of unsupervised learning in machine learning and AI, with a focus on Earth Observation (EO) scenarios. Rather than diving into theoretical details, our goal is to provide clear guidance and effective tools for applying these techniques in real-world classification tasks. Since unsupervised learning excels at recognizing patterns and structuring data without predefined labels, we will explore how these methods can uncover hidden relationships within datasets, enhancing our ability to categorize and analyze environmental features.

### Intro to Unsupervised Learning Methods

#### Introduction to K-means Clustering

K-means clustering is an unsupervised learning algorithm that partitions data into k predefined clusters based on feature similarity (MacQueen et al., 1967). It iteratively assigns data points to the nearest centroid and updates centroids to minimize within-cluster variance.

##### Why K-means?
Works well for exploratory analysis when data structure is unknown.
Simple & scalable for large datasets.
##### Key Components:
1. Choosing k – The number of clusters.
2. Centroid Initialization – Initial placement affects results.
3. Assignment Step – Points assigned to the nearest centroid.
4. Update Step – Centroids recalculated iteratively.

##### Advantages:
✔ Efficient and computationally fast.
✔ Easy to interpret and apply to real-world problems.
