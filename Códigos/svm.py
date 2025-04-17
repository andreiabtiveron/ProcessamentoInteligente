import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# Load dataset (Iris dataset as an example)
data = load_iris()
X = data.data
y = data.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM model (using linear kernel for feature importance)
svm = SVC(kernel='linear')
svm.fit(X_scaled, y)

# Use SelectFromModel to select important features
selector = SelectFromModel(svm, prefit=True, threshold='mean')  # Use 'mean' or a specific threshold
X_selected = selector.transform(X_scaled)

# Get selected feature indices
selected_features = selector.get_support(indices=True)

# Create a DataFrame to display selected features
feature_names = data.feature_names
selected_feature_names = [feature_names[i] for i in selected_features]
print("Selected features:", selected_feature_names)

# Optionally, display the selected features DataFrame
selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
print(selected_df.head())