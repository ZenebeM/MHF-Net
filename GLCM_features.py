import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure, feature

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

# Function to extract GLCM features from an image
def extract_glcm_features(image, folder_name):
    # Resize the image to 256x256
    resized_image = cv2.resize(image, (256, 256))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Compute GLCM in four directions with step size 1
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'correlation', 'energy', 'entropy']
    glcm_features = {}
    
    for angle in angles:
        glcm = feature.graycomatrix(gray, distances=distances, angles=[angle], symmetric=True, normed=True)
        for prop in properties:
            if prop not in glcm_features:
                glcm_features[prop] = []
            if prop == 'entropy':
                glcm_prop = feature.graycoprops(glcm, prop='contrast')[0, 0]
            else:
                glcm_prop = feature.graycoprops(glcm, prop=prop)[0, 0]
            glcm_features[prop].append(glcm_prop)
    
    # Create a dictionary with all GLCM features
    features = {
        'folder_name': folder_name,
        'contrast_0': glcm_features['contrast'][0],
        'contrast_45': glcm_features['contrast'][1],
        'contrast_90': glcm_features['contrast'][2],
        'contrast_135': glcm_features['contrast'][3],
        'dissimilarity_0': glcm_features['dissimilarity'][0],
        'dissimilarity_45': glcm_features['dissimilarity'][1],
        'dissimilarity_90': glcm_features['dissimilarity'][2],
        'dissimilarity_135': glcm_features['dissimilarity'][3],
        'homogeneity_0': glcm_features['homogeneity'][0],
        'homogeneity_45': glcm_features['homogeneity'][1],
        'homogeneity_90': glcm_features['homogeneity'][2],
        'homogeneity_135': glcm_features['homogeneity'][3],
        'correlation_0': glcm_features['correlation'][0],
        'correlation_45': glcm_features['correlation'][1],
        'correlation_90': glcm_features['correlation'][2],
        'correlation_135': glcm_features['correlation'][3],
        'energy_0': glcm_features['energy'][0],
        'energy_45': glcm_features['energy'][1],
        'energy_90': glcm_features['energy'][2],
        'energy_135': glcm_features['energy'][3],
        'entropy_0': glcm_features['entropy'][0],
        'entropy_45': glcm_features['entropy'][1],
        'entropy_90': glcm_features['entropy'][2],
        'entropy_135': glcm_features['entropy'][3]
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
            features = extract_glcm_features(image, subdir)
            all_features.append(features)

# Create a DataFrame and save to CSV
df = pd.DataFrame(all_features)
output_csv = "path/to/save/glcm_features.csv"
df.to_csv(output_csv, index=False)

print(f"GLCM Features saved to {output_csv}")
