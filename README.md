
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

##### Key Components:
1. Choosing k – The number of clusters.
2. Centroid Initialization – Initial placement affects results.
3. Assignment Step – Points assigned to the nearest centroid.
4. Update Step – Centroids recalculated iteratively.

#### Basic Code Implementation

```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```

![image](https://github.com/user-attachments/assets/78e9e2cd-be40-4658-b2e9-a01979d0fd27)

This image is the scatter plot showing the results of K-means clustering. The data points are grouped into different colored clusters, and the gray circles  represent the cluster centroids.

#### Introduction to Gaussian Mixture Models
Gaussian Mixture Models (GMM) are probabilistic models that represent data as a mixture of multiple Gaussian distributions, making them useful for clustering and density estimation (Bishop & Nasrabadi, 2006). Unlike K-means, GMM assigns probabilities to each data point belonging to multiple clusters, providing soft clustering and handling variations in cluster shape (Reynolds et al., 2009).

##### Key Components:
1. Number of Components – Equivalent to clusters in K-means.
2. Expectation-Maximization (EM) – Iterative algorithm to optimize Gaussian parameters.
3. Covariance Types – Controls the shape and spread of clusters (e.g., spherical, diagonal, full).

#### Basic Code Implementation
```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
```

![image](https://github.com/user-attachments/assets/d7d4ca5b-ef49-461b-9dd9-90fa21ab0518)

This image is the scatter plot showing the results of GMM clustering. The data points are grouped into different colored clusters, and the gray circles  represent the cluster centroids.

### Image Classification
Now, we apply these unsupervised learning techniques to image classification, specifically to differentiate between sea ice and leads in Sentinel-2 imagery. By leveraging clustering methods like K-means and Gaussian Mixture Models (GMM), we aim to uncover patterns in the data without requiring labeled samples, enabling automated and scalable classification of sea ice features.

#### K-Means Implementation
```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
```
![image](https://github.com/user-attachments/assets/10526cf2-3cc1-4910-9431-f6822ca0fe6f)

This visualization showcases K-means clustering applied to Sentinel-2 Band 4 imagery for distinguishing surface features like sea ice and leads. The script reads the Red Band (B4), applies a mask to filter valid pixels, and clusters the data into two groups (k=2). The dark purple region likely represents open water or shadows, while the teal region corresponds to sea ice, with yellow patches indicating distinct surface variations, possibly thinner ice or leads. 

#### GMM Implementation
```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
![image](https://github.com/user-attachments/assets/ffb7001a-f7f3-468a-9bfc-697f70a234ba)

This visualization represents Gaussian Mixture Model (GMM) clustering applied to Sentinel-2 Band 4 imagery for distinguishing sea ice and leads. The script reads the Red Band (B4), masks invalid pixels, and clusters the data into two components (n=2) using a probabilistic approach. Compared to K-means, GMM allows for soft clustering and adapts better to variations in data distribution. The dark purple region likely represents open water, while the teal and yellow regions capture variations in sea ice and leads. This method enhances segmentation accuracy by incorporating cluster uncertainty and flexible covariance structures. _This code needs longer time to plot the image, roughly around 3 minutes, so be patient._

### Altimetry Classification
Now, we apply unsupervised learning techniques to altimetry classification, specifically for distinguishing between sea ice and leads in the Sentinel-3 altimetry dataset. By leveraging clustering methods such as K-means and Gaussian Mixture Models (GMM), we aim to analyze waveform characteristics and height variations to automatically classify surface types, enhancing the accuracy of sea ice monitoring and lead detection.

#### Read in Functions Needed

Before diving into the modeling process, it is essential to preprocess the data to ensure compatibility with analytical models. This step involves transforming raw altimetry measurements into meaningful features, such as peakiness and stack standard deviation (SSD), which help capture key characteristics of the surface types. Proper preprocessing enhances the effectiveness of clustering methods in distinguishing sea ice and leads within the Sentinel-3 altimetry dataset. 

Since there are NaN values in the data, we need to remove them and then using the GMM model, plot the mean waveform of each class

```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```
![image](https://github.com/user-attachments/assets/1457f99d-a84a-4d83-88ff-a649afbb7209)

Blue line means sea ice and orange line means Lead

```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```
![image](https://github.com/user-attachments/assets/ee1963cd-9518-418f-abdb-133fcf22b08a)

```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```
![image](https://github.com/user-attachments/assets/722e1c4c-ecd3-40c5-aba7-25e3210be66c)

```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```
![image](https://github.com/user-attachments/assets/16e2c386-5894-46ef-940e-cb5520624fbf)

### Scatter Plots of Clustered Data
```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
```
![image](https://github.com/user-attachments/assets/d5878f72-23ed-41ba-89af-e8607b69f868)
![image](https://github.com/user-attachments/assets/1292e776-fadd-4858-b3d8-169bc2ad3217)
![image](https://github.com/user-attachments/assets/c66836e0-f989-495b-8638-616f78ba3419)

GMM classified Sentinel-3 altimetry data into distinct clusters. The scatter plots illustrate the relationships between sigma naught (σ₀), Peakiness Parameter (PP), and Stack Standard Deviation (SSD).

1. σ₀ vs. PP: Highlights differences in backscatter intensity and waveform peakiness.
2. σ₀ vs. SSD: Demonstrates variations in waveform deviation.
3. PP vs. SSD: Provides a strong separation between clusters, indicating that peakiness and waveform standard deviation are key indicators in classifying surface features.

These results confirm that unsupervised learning techniques, particularly GMM clustering, effectively segment altimetric features and enhance automated sea ice and lead classification. This approach strengthens remote sensing applications by providing probabilistic classifications that account for data uncertainty and variability in surface characteristics. 

### Waveform Alignment Using Cross-Correlation
```python
from scipy.signal import correlate
 
# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm==0], axis=0))
 
# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm==0][::len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm==0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)
    aligned_waves.append(aligned_wave)
 
# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)
 
plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
```
![image](https://github.com/user-attachments/assets/7a0520f0-1964-4847-8479-eb6d33debd74)

This process aligns waveforms within GMM cluster 0 to ensure peak synchronization, aiding in sea ice classification using Sentinel-3 altimetry data. The alignment is performed through cross-correlation, where the mean waveform of the cluster is computed, and its highest peak is identified as the reference point. Each waveform is then shifted using np.roll() to align with the reference, ensuring consistency. A subset of 10 equally spaced waveforms is plotted, revealing a sharp, consistent peak, which is characteristic of sea ice reflections. 

### Compare with ESA data
```python
flag_cleaned_modified = flag_cleaned - 1
from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned_modified   # true labels from the ESA dataset
predicted_gmm = clusters_gmm          # predicted labels from GMM method

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_labels, predicted_gmm)

# Print classification report
print("\nClassification Report:")
print(class_report)
```
```python
Confusion Matrix:
[[8856   22]
 [  24 3293]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195
```

In this data set, since sea ice = 1 and lead = 2, we need to minus 1 from flag_cleaned to maker sure predicted labels are comparable with the official product labels. The confusion matrix made a summary of data and classification report recorded the precision recall, f1-score and support, which showed a very high accuracy of 100%. With the number of 8856 of sea ice and 3293 of lead.
