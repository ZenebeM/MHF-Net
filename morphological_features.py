import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure

# Path to the parent folder containing subfolders with images
parent_folder = "patth/to/data/test" # during training, replace it with train data

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

# Function to extract morphological features from an image
def extract_morphological_features(image, folder_name):
    # Resize the image to 256x256
    resized_image = cv2.resize(image, (256, 256))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Label connected regions
    labeled_image = measure.label(binary, connectivity=2)
    
    # Measure region properties
    properties = measure.regionprops(labeled_image)
    
    # Initialize lists to collect feature values
    areas = []
    perimeters = []
    eccentricities = []
    equivalent_diameters = []
    extents = []
    solidities = []
    major_axis_lengths = []
    minor_axis_lengths = []
    roundnesses = []
    shape_factors = []
    compactnesses = []

    for prop in properties:
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        area = prop.area
        perimeter = prop.perimeter
        roundness = (4 * area * np.pi) / (perimeter ** 2) if perimeter != 0 else 0
        shape_factor = (perimeter ** 2) / area if area != 0 else 0
        compactness = (minor_axis_length / major_axis_length) if major_axis_length != 0 else 0

        areas.append(area)
        perimeters.append(perimeter)
        eccentricities.append(prop.eccentricity)
        equivalent_diameters.append(prop.equivalent_diameter)
        extents.append(prop.extent)
        solidities.append(prop.solidity)
        major_axis_lengths.append(major_axis_length)
        minor_axis_lengths.append(minor_axis_length)
        roundnesses.append(roundness)
        shape_factors.append(shape_factor)
        compactnesses.append(compactness)

    # Compute mean values for each feature
    feature_dict = {
        'folder_name': folder_name,
        'area': np.mean(areas) if areas else 0,
        'perimeter': np.mean(perimeters) if perimeters else 0,
        'eccentricity': np.mean(eccentricities) if eccentricities else 0,
        'equivalent_diameter': np.mean(equivalent_diameters) if equivalent_diameters else 0,
        'extent': np.mean(extents) if extents else 0,
        'solidity': np.mean(solidities) if solidities else 0,
        'major_axis_length': np.mean(major_axis_lengths) if major_axis_lengths else 0,
        'minor_axis_length': np.mean(minor_axis_lengths) if minor_axis_lengths else 0,
        'roundness': np.mean(roundnesses) if roundnesses else 0,
        'shape_factor': np.mean(shape_factors) if shape_factors else 0,
        'compactness': np.mean(compactnesses) if compactnesses else 0
    }

    return feature_dict

# List to store all features
all_features = []

# Walk through all subfolders and process images
for root, dirs, files in os.walk(parent_folder):
    for subdir in dirs:
        subfolder_path = os.path.join(root, subdir)
        images, filenames = load_images_from_folder(subfolder_path)
        for image, filename in zip(images, filenames):
            features = extract_morphological_features(image, subdir)
            all_features.append(features)

# Create a DataFrame and save to CSV
df = pd.DataFrame(all_features)
df.to_csv("path/to/save/morphological_features_test.csv", index=False)

print(f"Features saved to path/to/save/morphological_features_test.csv")
