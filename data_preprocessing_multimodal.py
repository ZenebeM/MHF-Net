"""
Preprocessing script for esophageal cancer diagnosis.
Handles clinical (tabular), deep, and handcrafted feature extraction.
"""
import os
import numpy as np
import pandas as pd
import cv2
from skimage import measure
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tqdm import tqdm


def preprocess_clinical_data(csv_path):
    """
    Load clinical tabular data and split into features and labels.
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def extract_deep_features(model: Model, directory: str, target_size=(224, 224)):
    """
    Run images through a CNN feature extractor to get deep features.
    """
    features, labels = [], []
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(directory, cls)
        for fname in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, fname)
            arr = load_and_preprocess_image(img_path, target_size)
            if arr is not None:
                feat = model.predict(arr)
                features.append(feat.flatten())
                labels.append(idx)
    return np.array(features), np.array(labels)


def extract_color_and_grayscale_features(img):
    rgb = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    feats = {
        'h_mean': np.mean(h), 's_mean': np.mean(s), 'v_mean': np.mean(v),
        'h_var': np.var(h),  's_var': np.var(s),  'v_var': np.var(v),
        'h_skew': skew(h.flatten()), 's_skew': skew(s.flatten()), 'v_skew': skew(v.flatten())
    }
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    feats.update({
        'gray_mean': np.mean(gray),
        'gray_var': np.var(gray),
        'gray_energy': np.sum(gray.astype(np.float64)**2)
    })
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    feats['gray_contrast'] = graycoprops(glcm, 'contrast')[0,0]
    return feats


def extract_glcm_features(img):
    rgb = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    dists = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props = ['contrast','dissimilarity','homogeneity','correlation','energy']
    out = {}
    for angle in angles:
        glcm = graycomatrix(gray, distances=dists, angles=[angle], symmetric=True, normed=True)
        for p in props:
            out[f"{p}_{int(np.degrees(angle))}"] = graycoprops(glcm, p)[0,0]
        entropy = -np.sum(glcm * np.log2(glcm + (glcm==0)))
        out[f"entropy_{int(np.degrees(angle))}"] = entropy
    return out


def extract_morphological_features(img):
    rgb = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)
    stats = {key: [] for key in ['area','perimeter','eccentricity','equiv_diameter',
                                  'extent','solidity','maj_axis','min_axis']}
    for prop in props:
        stats['area'].append(prop.area)
        stats['perimeter'].append(prop.perimeter)
        stats['eccentricity'].append(prop.eccentricity)
        stats['equiv_diameter'].append(prop.equivalent_diameter)
        stats['extent'].append(prop.extent)
        stats['solidity'].append(prop.solidity)
        stats['maj_axis'].append(prop.major_axis_length)
        stats['min_axis'].append(prop.minor_axis_length)
    out = {k: np.mean(v) if v else 0 for k,v in stats.items()}
    out['roundness'] = (4*np.pi*out['area'])/(out['perimeter']**2) if out['perimeter']>0 else 0
    return out


def extract_handcrafted_features(directory: str):
    """
    Extract HSV, GLCM, and morphological features for all images in a directory tree.
    """
    all_feats, labels = [], []
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(directory, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            img = cv2.imread(path)
            if img is None: continue
            feats = {}
            feats.update(extract_color_and_grayscale_features(img))
            feats.update(extract_glcm_features(img))
            feats.update(extract_morphological_features(img))
            all_feats.append(feats)
            labels.append(idx)
    df = pd.DataFrame(all_feats)
    df['label'] = labels
    return df


if __name__ == '__main__':
    # Paths
    train_dir = 'path/to/data/train'
    test_dir  = 'path/to/data/test'
    clinical_csv = 'path/to/clinical_data/esophageal.csv'

    # 1. Clinical data
    X_clin, y_clin = preprocess_clinical_data(clinical_csv)
    clinical_df = pd.concat([X_clin, y_clin.rename('label')], axis=1)
    clinical_df.to_csv('clinical_features.csv', index=False)

    # 2. Deep features (using a pre-trained MobileNetV2 backbone)
    base = image.load_img  # Placeholder: load your saved feature_extractor_model
    from tensorflow.keras.applications import MobileNetV2
    backbone = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model = Model(inputs=backbone.input, outputs=GlobalAveragePooling2D()(backbone.output))

    Xd_train, yd_train = extract_deep_features(model, train_dir)
    Xd_test,  yd_test  = extract_deep_features(model, test_dir)
    deep_df = pd.DataFrame(np.vstack([Xd_train, Xd_test]))
    deep_df['label'] = np.concatenate([yd_train, yd_test])
    deep_df.to_csv('deep_features.csv', index=False)

    # 3. Handcrafted features
    hf_train = extract_handcrafted_features(train_dir)
    hf_test  = extract_handcrafted_features(test_dir)
    handcrafted_df = pd.concat([hf_train, hf_test], ignore_index=True)
    handcrafted_df.to_csv('handcrafted_features.csv', index=False)

    # 4. Combined features
    combined = pd.concat([
        handcrafted_df.drop(columns=['label']),
        deep_df.drop(columns=['label']),
        clinical_df['label']], axis=1)
    combined.to_csv('combined_all_features.csv', index=False)

    print("Preprocessing complete. CSV files saved.")
