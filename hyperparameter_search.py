import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("path/to/load/selected_DHF.csv")


# Check for missing values and handle them (if any)
if data.isnull().sum().sum() > 0:
    data = data.dropna()  # Alternatively, you could use imputation methods

# Preprocess the data
# Assuming the target variable is named 'target' and is the last column
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode categorical variables if necessary
# Example: If the target variable is categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# If there are categorical features in X, encode them as well
X = pd.get_dummies(X)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=50, random_state=42)
#fir the model
rf_model.fit(X_train, y_train)
# Make predictions with both models
rf_predictions = rf_model.predict(X_test)
# Evaluate the models
print("\nRandom Forest Classifier:")
print("Accuracy on test set:", accuracy_score(y_test, rf_predictions))
#print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Initialize the XGBoost model
xgb_model = XGBClassifier(colsample_bytree=0.9, learning_rate=0.1, max_depth=12, n_estimators=100, subsample=0.8, random_state=42,  eval_metric='mlogloss')
# Fit the models on the training data
xgb_model.fit(X_train, y_train)
#Make predictions
xgb_predictions = xgb_model.predict(X_test)
#Evaluate the model
print("\nXGBoost Classifier:")
print("Accuracy on test set:", accuracy_score(y_test, xgb_predictions))
#print("Classification Report:\n", classification_report(y_test, xgb_predictions))

# Define the labels explicitly
labels = np.arange(7)  # Assuming the class labels are from 0 to 14

# Plot confusion matrices
rf_cm = confusion_matrix(y_test, rf_predictions, labels=labels)
xgb_cm = confusion_matrix(y_test, xgb_predictions, labels=labels)


fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ConfusionMatrixDisplay(rf_cm, display_labels=labels).plot(ax=ax[0], cmap=plt.cm.Blues)
ax[0].set_title('Random Forest Confusion Matrix')
ConfusionMatrixDisplay(xgb_cm, display_labels=labels).plot(ax=ax[1], cmap=plt.cm.Blues)
ax[1].set_title('XGBoost Confusion Matrix')
plt.show()

