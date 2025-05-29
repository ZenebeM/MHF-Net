import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the parent folder containing subfolders with images
parent_folder = "path/to/data/"

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(img_path)
    return images, filenames

# Function to extract color and grayscale features from an image
def extract_features(image, folder_name):
    # Resize the image to 256x256
    resized_image = cv2.resize(image, (256, 256))
    
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    
    # Split HSV channels
    h, s, v = cv2.split(hsv_image)
    
    # Calculate first moment (mean), second moment (variance), and third-order moment (skewness) for HSV channels
    hsv_features = {
        'h_mean': np.mean(h),
        's_mean': np.mean(s),
        'v_mean': np.mean(v),
        'h_variance': np.var(h),
        's_variance': np.var(s),
        'v_variance': np.var(v),
        'h_skewness': skew(h.flatten()),
        's_skewness': skew(s.flatten()),
        'v_skewness': skew(v.flatten())
    }
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale features: mean, variance, energy, and contrast
    gray_mean = np.mean(gray_image)
    gray_variance = np.var(gray_image)
    gray_energy = np.sum(gray_image ** 2)
    
    # Calculate GLCM and contrast
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    gray_contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    grayscale_features = {
        'gray_mean': gray_mean,
        'gray_variance': gray_variance,
        'gray_energy': gray_energy,
        'gray_contrast': gray_contrast
    }
    
    # Combine features into a single dictionary
    features = {
        'folder_name': folder_name,
        **hsv_features,
        **grayscale_features
    }
    
    return features

# List to store all features
all_features = []

# Walk through all subfolders and process images
for root, dirs, files in os.walk(parent_folder):
    for subdir in dirs:
        subfolder_path = os.path.join(root, subdir)
        images, filenames = load_images_from_folder(subfolder_path)
        for image, filename in zip(images, filenames):
            features = extract_features(image, subdir)
            all_features.append(features)

# Create a DataFrame and save to CSV
df = pd.DataFrame(all_features)
output_csv = "path/to/save/color_grayscale_features.csv"
df.to_csv(output_csv, index=False)

print(f"Features saved to {output_csv}")

# Exploratory Data Analysis (EDA)

# Load the extracted features
df = pd.read_csv(output_csv)

# Display the first few rows of the DataFrame
print(df.head())

# Display summary statistics
print(df.describe())

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Plot histograms for each feature
df.hist(bins=30, figsize=(15, 10), layout=(6, 3))
plt.suptitle('Feature Distributions')
plt.show()

# Plot density plots for each feature
df.plot(kind='density', subplots=True, layout=(6, 3), sharex=False, figsize=(15, 10))
plt.suptitle('Feature Density Plots')
plt.show()

# Plot box plots for each feature
plt.figure(figsize=(15, 10))
sns.boxplot(data=df, orient='h')
plt.title('Box Plots of Features')
plt.show()

# Plot pairwise relationships between features
sns.pairplot(df)
plt.suptitle('Pairwise Relationships of Features', y=1.02)
plt.show()

# Example validation of HSV mean values
print("HSV Mean Values:")
print(df[['h_mean', 's_mean', 'v_mean']].head())

# Example validation of grayscale mean and variance
print("Grayscale Mean and Variance:")
print(df[['gray_mean', 'gray_variance']].head())
