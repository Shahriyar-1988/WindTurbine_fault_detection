import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import shap
import numpy as np
from keras.models import load_model
from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from src.utils.common import read_yaml
import pandas as pd
import matplotlib.pyplot as plt

# model related paths
file_path = read_yaml(CONFIG_FILE_PATH)["classifier_evaluation"]
model_path=file_path["cls_model_path"]
data_path=file_path["test_data_paths"][0]


model=load_model(model_path)
# Load model and data
X_data = np.load(data_path)  # shape: (N, 78, 3) or (N, 234)
# Confirm the shape is 3D
assert X_data.ndim == 3, f"Expected 3D input, got {X_data.shape}"

# Background and sample data (keep 3D shape)
X_background = X_data[:200]
X_sample = X_data[:50]

# Create SHAP explainer
explainer = shap.DeepExplainer(model, X_background)
shap_values = explainer(X_sample)

# Load feature names
data_dir=read_yaml(CONFIG_FILE_PATH)["classifier_training"]
schema=read_yaml(SCHEMA_FILE_PATH)
target_col=schema["TARGET_COLUMN"]
data_path=data_dir["data_path"]
feature_names_ = pd.read_csv(data_path).drop(columns=[target_col]).columns.tolist()

# Flatten for visualization (shap_values and X_sample)
shap_vals_flat = shap_values.values.reshape(-1, X_sample.shape[1] * X_sample.shape[2])
X_flat = X_sample.reshape(-1, X_sample.shape[1] * X_sample.shape[2])

# Save SHAP plot
save_dir = "src/Analysis"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "shap_summary_plot.png")

plt.figure()
shap.summary_plot(shap_vals_flat, X_flat, feature_names=feature_names_, show=False)
plt.savefig(save_path, bbox_inches='tight')
plt.close()
 ## Bar plot for top 5 features
# ======================

# Mean absolute SHAP values per feature
mean_shap = np.abs(shap_vals_flat).mean(axis=0)
top5_idx = np.argsort(mean_shap)[-5:][::-1]  # indices of top 5
top5_names = [feature_names_[i] for i in top5_idx]
top5_values = mean_shap[top5_idx]

# Plot
plt.figure(figsize=(8, 5))
plt.barh(top5_names[::-1], top5_values[::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("Top 5 Important Features (SHAP)")
plt.tight_layout()

# Save
bar_path = os.path.join(save_dir, "shap_top5_bar_plot.png")
plt.savefig(bar_path)
plt.close()