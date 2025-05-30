import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np 
import cv2
import pandas as pd
from skimage import measure
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             classification_report, roc_curve, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121
#from tensorflow.keras.applications.resnet import Resnet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Concatenate, Dense, Input, GlobalAveragePooling2D, Multiply, Reshape, Lambda
from tensorflow.keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.layers import (Conv2D, Lambda,MaxPooling2D, Flatten, Dense, Dropout, 
                                     Input, BatchNormalization, Activation, 
                                     GlobalAveragePooling2D, GlobalMaxPooling2D, 
                                     Reshape, Multiply, Add, Concatenate)
 

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# training and testing dataset
train_dir = "path/to/train/data"
test_dir = "path/to/test/data"

# clinical/esophageal dataset
esophageal = pd.read_csv("path/to/clinical/data/esophageal_data.csv")
X_esophageal = esophageal.drop(columns=['label'])
y_esophageal = esophageal['label']

# Function to load and preprocess images
def load_and_preprocess_image(img_path, target_size=(256, 256)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None
   
# Function to extract deep features from a directory
def extract_deep_features_from_directory(model, directory, target_size=(256, 256)):
    features = []
    labels = []
    class_labels = sorted(os.listdir(directory))
    for label in tqdm(class_labels, desc="Extracting deep features"):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = load_and_preprocess_image(img_path, target_size)
                if img is not None:
                    feature = model.predict(img)
                    features.append(feature.flatten())  # Flatten the feature array
                    labels.append(class_labels.index(label))
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

# Define the CBAM block
def cbam_block(cbam_feature, ratio=8):
    # Channel attention
    channel = GlobalAveragePooling2D()(cbam_feature)
    channel = Reshape((1, 1, cbam_feature.shape[-1]))(channel)
    channel = Dense(cbam_feature.shape[-1] // ratio, activation='relu')(channel)
    channel = Dense(cbam_feature.shape[-1], activation='sigmoid')(channel)
    channel_feature = Multiply()([cbam_feature, channel])
    # Spatial attention
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    spatial = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    spatial_feature = Multiply()([channel_feature, spatial])
    return spatial_feature

# Function to extract color and grayscale features from an image
def extract_color_and_grayscale_features(image):
    # Resize 
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
    # Calculate GLCM features
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = {
        'gray_contrast': graycoprops(glcm, 'contrast')[0, 0]
    }   
    grayscale_features = {
        'gray_mean': gray_mean,
        'gray_variance': gray_variance,
        'gray_energy': gray_energy,
        **glcm_features
    }   
    # Combine features into a single dictionary
    features = {**hsv_features, **grayscale_features}   
    return features

def extract_glcm_features(image):
    # Resize 
    resized_image = cv2.resize(image, (256, 256))
    # image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Compute GLCM in four directions with step size 1
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'correlation', 'energy']
    glcm_features = {}
    
    for angle in angles:
        glcm = graycomatrix(gray, distances=distances, angles=[angle], symmetric=True, normed=True)
        for prop in properties:
            if prop not in glcm_features:
                glcm_features[prop] = []
            glcm_prop = graycoprops(glcm, prop=prop)[0, 0]
            glcm_features[prop].append(glcm_prop) 
        # Calculate entropy separately
        if 'entropy' not in glcm_features:
            glcm_features['entropy'] = []
        entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
        glcm_features['entropy'].append(entropy)
    features = {
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

def extract_morphological_features(image):
    # Resize 
    resized_image = cv2.resize(image, (256, 256))    
    #image to grayscale
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

def extract_all_features(image, folder_name):
    color_and_grayscale_features = extract_color_and_grayscale_features(image)
    glcm_features = extract_glcm_features(image)
    morphological_features = extract_morphological_features(image)
    features = {**color_and_grayscale_features, **glcm_features, **morphological_features}
    return features

def extract_features_from_directory(directory):
    features_list = []
    class_labels = sorted(os.listdir(directory))
    for label in tqdm(class_labels, desc=f"Extracting features from {directory}"):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        features = extract_all_features(image, label)
                        features_list.append(features)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    if features_list:
        df = pd.DataFrame(features_list)
    else:
        df = pd.DataFrame()  # Empty DataFrame if no features extracted
    return df

def get_class_labels(train_dir):
    class_labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return class_labels

class GrasshopperOptimizer:
    def __init__(self, obj_function, lb, ub, dim, X, y, n_grasshoppers=20, max_iter=100):
        self.obj_function = obj_function
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.X = X
        self.y = y
        self.n_grasshoppers = n_grasshoppers
        self.max_iter = max_iter
        self.positions = np.random.uniform(lb, ub, (n_grasshoppers, dim))
        self.best_position = None
        self.best_score = float('inf')

    def optimize(self):
        for t in range(self.max_iter):
            # Calculate fitness for each grasshopper
            fitness = np.apply_along_axis(lambda pos: self.obj_function(pos, self.X, self.y), 1, self.positions)

            # Update best position and score
            for i in range(self.n_grasshoppers):
                if fitness[i] < self.best_score:
                    self.best_score = fitness[i]
                    self.best_position = self.positions[i, :]

            # Update positions of grasshoppers based on their interactions
            c = 0.5 - (t / (2 * self.max_iter))  # Decreasing coefficient
            new_positions = np.zeros_like(self.positions)

            for i in range(self.n_grasshoppers):
                temp_position = np.zeros(self.dim)
                for j in range(self.n_grasshoppers):
                    if i != j:
                        distance = np.linalg.norm(self.positions[j] - self.positions[i])
                        s = (self.positions[j] - self.positions[i]) / (distance + 1e-10) * np.exp(-distance)
                        temp_position += s

                new_positions[i] = c * temp_position + self.positions[i]

            # Ensure grasshoppers stay within bounds
            self.positions = np.clip(new_positions, self.lb, self.ub)

        return self.best_position, self.best_score

# Objective function for GOA
def objective_function_rf(params, X, y):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=False,
        random_state=42
    )
    
    accuracies = []
    for _ in range(3):  # Perform 3 runs for each configuration to get a stable estimate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=np.random.randint(0, 10000))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    # Objective is to maximize accuracy, so we return the negative mean accuracy
    return -np.mean(accuracies)

def objective_function_xgb(params, X, y):
    learning_rate = params[0]
    max_depth = int(params[1])
    n_estimators = int(params[2])
    subsample = params[3]
    colsample_bytree = params[4]

    clf = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        eval_metric='mlogloss',
        objective='multi:softmax'
    )
    
    accuracies = []
    for _ in range(3):  # Perform 3 runs for each configuration to get a stable estimate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=np.random.randint(0, 10000))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    # Objective is to maximize accuracy, so we return the negative mean accuracy
    return -np.mean(accuracies)

def repeated_random_forest_with_goa(X, y, class_names, n_runs=100, test_size=0.15,
                                    output_file_cm='confusion_matrix.jpg', report_file='classification_report.csv',
                                    history_file='training_history.csv', output_file_roc='roc_curves.jpg',
                                    n_grasshoppers=10, max_iter=100): #n_runs=100
    
    '''
    # Define bounds for hyperparameters: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    lb = [50, 5, 2, 1]
    ub = [200, 20, 10, 5]

    # Run GOA optimization
    goa = GrasshopperOptimizer(objective_function_rf, lb, ub, dim=4, X=X, y=y, n_grasshoppers=n_grasshoppers, max_iter=max_iter)
    best_params, best_score = goa.optimize()

    # Extract optimized hyperparameters
    n_estimators = int(best_params[0])
    max_depth = int(best_params[1])
    min_samples_split = int(best_params[2])
    min_samples_leaf = int(best_params[3])

    print(f'Optimized Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, '
          f'min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}')
    
   ''' 

    # Now run the repeated Random Forest with the optimized hyperparameters
    scores = []
    unique_classes = np.unique(y)
    cm_total = np.zeros((len(unique_classes), len(unique_classes)))
    history = {'run': [], 'accuracy': []}

    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_fprs = [[] for _ in range(len(unique_classes))]
    roc_tprs = [[] for _ in range(len(unique_classes))]
    roc_auc_scores = [[] for _ in range(len(unique_classes))]

    all_y_true = []
    all_y_pred = []

    for i in range(n_runs):
        random_state = np.random.randint(0, 10000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        clf = RandomForestClassifier(n_estimators=115, max_depth=14, min_samples_split=2,
                                     min_samples_leaf=1, bootstrap=False, random_state=random_state)#min_samples_split=8
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        history['run'].append(i + 1)
        history['accuracy'].append(accuracy)

        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        cm_total += cm

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        for j in range(len(unique_classes)):
            y_test_binary = (y_test == unique_classes[j]).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_prob[:, j])
            roc_fprs[j].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            roc_tprs[j].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            roc_auc_scores[j].append(roc_auc_score(y_test_binary, y_prob[:, j]))

        print(f'Run {i+1}/{n_runs} - Accuracy: {accuracy:.4f}')

    cm_average = cm_total / n_runs

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_average, annot=True, fmt='.0f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Average Confusion Matrix')
    plt.savefig(output_file_cm, format='jpg')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    mean_fpr = np.linspace(0, 1, 100)
    for j in range(len(unique_classes)):
        mean_tpr = np.mean(roc_tprs[j], axis=0)
        mean_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)
        auc_score = np.mean(roc_auc_scores[j])
        plt.plot(mean_fpr, mean_tpr, label=f'{class_names[j]} (AUC = {auc_score:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curves')
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(output_file_roc, format='jpg')
    plt.close()

    report = classification_report(all_y_true, all_y_pred, labels=unique_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(report_file)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_file, index=False)

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    return scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df


def repeated_xgb_with_goa(X, y, class_names, n_runs=100, test_size=0.15,
                           output_file_cm='confusion_matrix.jpg', report_file='classification_report.csv',
                           history_file='training_history.csv', output_file_roc='roc_curves.jpg',
                           n_grasshoppers=10, max_iter=50): #n_runs=100
    '''
    # Define bounds for hyperparameters: [learning_rate, max_depth, n_estimators, subsample, colsample_bytree]
    lb = [0.01, 3, 50, 0.5, 0.5]
    ub = [0.3, 20, 300, 1.0, 1.0]

    # Run GOA optimization
    goa = GrasshopperOptimizer(objective_function_xgb, lb, ub, dim=5, X=X, y=y, n_grasshoppers=n_grasshoppers, max_iter=max_iter)
    best_params, best_score = goa.optimize()

    # Extract optimized hyperparameters
    learning_rate = best_params[0]
    max_depth = int(best_params[1])
    n_estimators = int(best_params[2])
    subsample = best_params[3]
    colsample_bytree = best_params[4]

    print(f'Optimized Hyperparameters: learning_rate={learning_rate}, max_depth={max_depth}, '
          f'n_estimators={n_estimators}, subsample={subsample}, colsample_bytree={colsample_bytree}')
    '''
    # Now run the repeated XGBoost with the optimized hyperparameters
    scores = []
    unique_classes = np.unique(y)
    cm_total = np.zeros((len(unique_classes), len(unique_classes)))
    history = {'run': [], 'accuracy': []}

    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_fprs = [[] for _ in range(len(unique_classes))]
    roc_tprs = [[] for _ in range(len(unique_classes))]
    roc_auc_scores = [[] for _ in range(len(unique_classes))]

    all_y_true = []
    all_y_pred = []

    for i in range(n_runs):
        random_state = np.random.randint(0, 10000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        clf = XGBClassifier(
            learning_rate=0.1,#0.1
            max_depth=14,
            n_estimators=115,
            subsample=0.7,
            colsample_bytree=0.5,
            random_state=random_state,
            eval_metric='mlogloss',
            objective='multi:softmax'
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        history['run'].append(i + 1)
        history['accuracy'].append(accuracy)

        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        cm_total += cm

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        for j in range(len(unique_classes)):
            y_test_binary = (y_test == unique_classes[j]).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_prob[:, j])
            roc_fprs[j].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            roc_tprs[j].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            roc_auc_scores[j].append(roc_auc_score(y_test_binary, y_prob[:, j]))

        print(f'Run {i+1}/{n_runs} - Accuracy: {accuracy:.4f}')

    cm_average = cm_total / n_runs

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_average, annot=True, fmt='.0f', cmap='Blues', xticklabels=class_names, yticklabels=class_names) #cmap='Blues'
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Average Confusion Matrix')
    plt.savefig(output_file_cm, format='jpg')
    plt.close()

    plt.figure(figsize=(10, 8)) 
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    mean_fpr = np.linspace(0, 1, 100)
    for j in range(len(unique_classes)):
        mean_tpr = np.mean(roc_tprs[j], axis=0)
        mean_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)
        auc_score = np.mean(roc_auc_scores[j])
        plt.plot(mean_fpr, mean_tpr, label=f'{class_names[j]} (AUC = {auc_score:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curves')
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(output_file_roc, format='jpg')
    plt.close()

    report = classification_report(all_y_true, all_y_pred, labels=unique_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(report_file)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_file, index=False)

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    return scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df

class_labels = get_class_labels(train_dir)
#print("Class Labels:", class_labels)

# Dilated Inception Block
def dilated_inception_block(x, filters):
    branch1 = Conv2D(filters, (1, 1), padding='same', name = "branch1", activation='relu')(x)
    branch2 = Conv2D(filters, (3, 3), padding='same', dilation_rate=1, name = "branch2", activation='relu')(branch1)
    branch3 = Conv2D(filters, (3, 3), padding='same', dilation_rate=2, name = "branch3",activation='relu')(branch1)
    branch4 = Conv2D(filters, (3, 3), padding='same', dilation_rate=3, name = "branch4",activation='relu')(branch1)
    branch5 = Conv2D(filters, (3, 3), padding='same', dilation_rate=3, name = "branch5",activation='relu')(branch2)
    output = Concatenate()([branch2, branch3, branch4,branch5])
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


# AlexNet (up to the fifth max-pooling layer)
def alexnet_base(input_tensor):
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='valid')(input_tensor)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x
'''
########################New Model#########################Start
# Complete model
input_layer = Input(shape=(256, 256, 3))
x = dilated_inception_block(input_layer, 64)
x = alexnet_base(x)
x = cbam_block(x)
feature_extractor_model = Model(inputs=input_layer, outputs=x)
feature_extractor_model.summary()    
# Save the model plot
plot_model(feature_extractor_model, to_file='path/to/save/model/DinceptionAlexnet-CBAM_model.jpg', show_shapes=True, show_layer_names=True)
########################New Model#########################End
'''

##---------------------------Deep Feature Extractor model--------------Start
mobilenet_base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
mobilenet_base_model.trainable = False
densenet_base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
densenet_base_model.trainable = False
input_layer = Input(shape=(256, 256, 3))
mobilenet_features = mobilenet_base_model(input_layer)
densenet_features = densenet_base_model(input_layer)
concatenated_features = Concatenate()([mobilenet_features, densenet_features])
conv_layer_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(concatenated_features)
cbam_features = cbam_block(conv_layer_1)
#conv_pooled_features = GlobalAveragePooling2D()(cbam_features)
conv_layer_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(cbam_features)
#max_pool = MaxPooling2D((2,2),name='max_pool_conv2')(conv_layer_2)
fc = Dense(512, activation = 'relu')(conv_layer_2)
gap = GlobalAveragePooling2D()(fc)

feature_extractor_model = Model(inputs=input_layer, outputs=gap)
feature_extractor_model.summary()

# Save the model plot
plot_model(feature_extractor_model, to_file='path/to/save/model/architecture/DenseMobileNet-CBAM_model.jpg', show_shapes=True, show_layer_names=True)
##---------------------------Deep Feature Extractor model--------------End

##-------------- deep feature -----------------------Start
train_deep_features, train_labels = extract_deep_features_from_directory(feature_extractor_model, train_dir)
test_deep_features, test_labels = extract_deep_features_from_directory(feature_extractor_model, test_dir)

combined_deep_features = np.concatenate([train_deep_features, test_deep_features], axis=0)
combined_deep_labels = np.concatenate([train_labels, test_labels], axis=0)

combined_deep_features = pd.DataFrame(combined_deep_features)
combined_deep_labels = pd.DataFrame(combined_deep_labels, columns=['label'])
combined_deep_dataset = pd.concat([combined_deep_features,combined_deep_labels],axis=1)

X_deep = combined_deep_dataset.drop(columns=['label'])
y_deep = combined_deep_dataset['label']

##Applying ML RF and XGB
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df = repeated_random_forest_with_goa(
    X_deep, 
    y_deep, 
    class_labels,
    n_runs=100, #n_runs=100
    output_file_cm='path/to/save/rf_confusion_matrix_deep_features.jpg', 
    report_file='path/to/save/rf_classification_report_deep_features.csv',
    history_file='path/to/save/rf_training_history_deep_features.csv',
    output_file_roc ='path/to/save/rf_roc_auc_curves_deep_features.jpg'
    
) 
print(f'Mean Accuracy RF DF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy RF DF: {std_accuracy:.4f}')
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df = repeated_xgb_with_goa(
    X_deep, 
    y_deep, 
    class_labels,
    n_runs=100, #n_runs=100
    output_file_cm='path/to/save/xgb_confusion_matrix_deep_features.jpg', 
    report_file='path/to/save/xgb_classification_report_deep_features.csv',
    history_file='path/to/save/xgb_training_history_deep_features.csv',
    output_file_roc ='path/to/save/xgb_roc_auc_curves_deep_features.jpg'
)
print(f'Mean Accuracy XGB DF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy XGB DF: {std_accuracy:.4f}')
##-------------- deep feature -----------------------End

##-------------- handcrafted feature-----------------------Start
train_handcrafted_features = extract_features_from_directory(train_dir)
test_handcrafted_features = extract_features_from_directory(test_dir)

combined_handcrafted_features = np.concatenate([train_handcrafted_features, test_handcrafted_features], axis=0)
combined_handcrafted_labels = np.concatenate([train_labels, test_labels], axis=0)

combined_handcrafted_features = pd.DataFrame(combined_handcrafted_features)
combined_handcrafted_labels = pd.DataFrame(combined_handcrafted_labels, columns=['label'])
combined_handcrafted_dataset = pd.concat([combined_handcrafted_features,combined_handcrafted_labels],axis=1)

X_handcraft = combined_handcrafted_dataset.drop(columns=['label'])
y_handcraft = combined_handcrafted_dataset['label']

##Applying ML RF and XGB
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df = repeated_random_forest_with_goa(
    X_handcraft, 
    y_handcraft, 
    class_labels,
    n_runs=100, #n_runs=100
    output_file_cm='path/to/save/rf_confusion_matrix_handcraft_features.jpg', 
    report_file='path/to/save/rf_classification_report_handcraft_features.csv',
    history_file='path/to/save/rf_training_history_handcraft_features.csv',
    output_file_roc ='path/to/save/rf_roc_auc_curves_handcraft_features.jpg'
)
print(f'Mean Accuracy RF HF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy RF HF: {std_accuracy:.4f}')
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df = repeated_xgb_with_goa(
    X_handcraft, 
    y_handcraft, 
    class_labels,
    n_runs=100, #n_runs=100
    output_file_cm='path/to/save/xgb_confusion_matrix_handcraft_features.jpg', 
    report_file='path/to/save/xgb_classification_report_handcraft_features.csv',
    history_file='path/to/save/xgb_training_history_handcraft_features.csv',
    output_file_roc ='path/to/save/xgb_roc_auc_curves_handcraft_features.jpg'
)
print(f'Mean Accuracy XGB HF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy XGB HF: {std_accuracy:.4f}')
##-------------- handcrafted feature----------------------End 

##-------------- combined(deep + handcrafted) features-----------------------Start
num_handcrafted_features = combined_handcrafted_features.shape[1]
num_deep_features = combined_deep_features.shape[1]
#num_esophageal = X_esophageal.shape[1]
num_labels = combined_handcrafted_labels.shape[1]
# Generate new column names
handcrafted_feature_names = [f'handcrafted_f_{i+1}' for i in range(num_handcrafted_features)]
deep_feature_names = [f'deep_f_{i+1}' for i in range(num_deep_features)]
#esophageal_feature_names = [f'met_f_{i+1}' for i in range(num_esophageal)]
label_names = [f'label_{i+1}' for i in range(num_labels)]

combined_deep_handcraft = pd.concat([combined_handcrafted_features,combined_deep_features,combined_handcrafted_labels],axis=1)
new_column_names = handcrafted_feature_names + deep_feature_names + label_names
combined_deep_handcraft.columns = new_column_names

X_combined = combined_deep_handcraft.drop(columns=['label_1'])
y_combined = combined_deep_handcraft['label_1']
##Applying ML RF and XGB
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df= repeated_random_forest_with_goa(
    X_combined, 
    y_combined, 
    class_labels,
    n_runs=100, #n_runs=100
    output_file_cm='path/to/save/rf_confusion_matrix_deep_handcraft_features.jpg', 
    report_file='path/to/save/rf_classification_report_deep_handcraft_features.csv',
    history_file='path/to/save/rf_training_history_deep_handcraft_features.csv',
    output_file_roc ='path/to/save/rf_roc_auc_curves_deep_handcraft_features.jpg'
)
print(f'Mean Accuracy RF DHF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy RF DHF: {std_accuracy:.4f}')
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df = repeated_xgb_with_goa(
    X_combined, 
    y_combined, 
    class_labels,
    n_runs=100, #n_runs=100
    output_file_cm='path/to/save/xgb_confusion_matrix_deep_handcraft_features.jpg', 
    report_file='path/to/save/xgb_classification_report_deep_handcraft_features.csv',
    history_file='path/to/save/xgb_training_history_deep_handcraft_features.csv',
    output_file_roc='path/to/save/xgb_roc_auc_curves_deep_handcraft_features.jpg'
)
print(f'Mean Accuracy XGB DHF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy XGB DHF: {std_accuracy:.4f}')
##-------------- combined(deep + handcrafted) eatures--------------End

##-------------- combined(deep + handcrafted + esophageal)-----------------------Start
combined_deep_handcraft_esophageal = pd.concat([combined_handcrafted_features,combined_deep_features,X_esophageal,combined_handcrafted_labels],axis=1)
#new_column_names_esophageal = handcrafted_feature_names + deep_feature_names + esophageal_feature_names + label_names
#combined_deep_handcraft_esophageal.columns = new_column_names_esophageal

X_combined_esophageal = combined_deep_handcraft_esophageal.drop(columns=['label_1'])
y_combined_esophageal = combined_deep_handcraft_esophageal['label_1']
##Applying ML RF and XGB
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df = repeated_random_forest_with_goa(
    X_combined_esophageal, 
    y_combined_esophageal, 
    class_labels,
    n_runs=100, 
    output_file_cm='path/to/save/rf_confusion_matrix_deep_handcraft_met_features.jpg',  
    report_file='path/to/save/rf_classification_report_deep_handcraft_met_features.csv',
    history_file='path/to/save/rf_training_history_deep_handcraft_met_features.csv',
    output_file_roc='path/to/save/rf_roc_auc_curves_deep_handcraft_met_features.jpg'
)
print(f'Mean Accuracy RF DHMF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy RF DHMF: {std_accuracy:.4f}')
scores, mean_accuracy, std_accuracy, cm_average, report_df, history_df = repeated_xgb_with_goa(
    X_combined_esophageal, 
    y_combined_esophageal, 
    class_labels,
    n_runs=100, 
    output_file_cm='path/to/save/xgb_confusion_matrix_deep_handcraft_met_features.jpg', 
    report_file='path/to/save/xgb_classification_report_deep_handcraft_met_features.csv',
    history_file='path/to/save/xgb_training_history_deep_handcraft_met_features.csv',
    output_file_roc='path/to/save/xgb_roc_auc_curves_deep_handcraft_met_features.jpg'
)
print(f'Mean Accuracy XGB DHMF: {mean_accuracy:.4f}')
print(f'Standard Deviation of Accuracy XGB DHMF: {std_accuracy:.4f}')
##-------------- combined(deep + handcrafted + esophageal)-----------------------End

##---------------Saving the dataset to CSV---------------------start
output_path = "path/to/save/handcrafted_features.csv"
combined_handcrafted_dataset.to_csv(output_path, index=False)
output_path = "path/to/save/deep_features.csv"
combined_deep_dataset.to_csv(output_path, index=False)
output_path = "path/to/save/deep_handcraft_features.csv"
combined_deep_handcraft.to_csv(output_path, index=False)
output_path = "path/to/save/deep_handcraft_esophageal_features.csv"
combined_deep_handcraft_esophageal.to_csv(output_path, index=False)
##---------------Saving the dataset to CSV---------------------End










