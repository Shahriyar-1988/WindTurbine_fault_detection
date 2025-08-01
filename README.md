# 🌀 Wind Turbine Fault Detection with Deep Learning

> A machine learning–assisted framework for early fault detection in operational wind turbines using SCADA data.

## 📌 Project Overview

This project was developed in collaboration with the **Renewable Energy Research Group** under the supervision of **Dr. Ahmaid**. The primary objective was to build an intelligent system that detects early-stage faults in wind turbines to optimize **inspection schedules** and **maintenance operations**.

We leveraged **SCADA (Supervisory Control and Data Acquisition)** sensor data from wind turbines, focusing on minimal supervision and interpretability.

---

## ⚙️ Key Features

- ✅ **Unsupervised Pretraining**: 1D convolutional autoencoder trained solely on _normal_ turbine data to learn healthy operating patterns.
- 🚀 **Transfer Learning**: Latent representation from the encoder is used to train a classifier.
- 📊 **Classification Performance**:  
  - **Test Accuracy**: ~80%  
  - **F1 Score**: ~85%
- 🧠 **SHAP Feature Importance**: Identifies critical sensor readings influencing model decisions.
- ⚖️ **Class Imbalance Handling**:  
  - The dataset was ~90:10 (normal:faulty).  
  - We used **ADASYN** oversampling to rebalance training data effectively.

---

## 🗂️ Repository Structure

```bash
WindTurbine_fault_detection/
│
├── ClassifierHead/               # Classifier built on top of encoder
├── EncoderBlock/                 # Autoencoder model and training
├── Notebooks/                    # Jupyter notebooks for exploration
├── artifacts/                    # Saved models, logs, and metrics
├── config/                       # YAML config files for schema and parameters
├── src/                          # Core source code
│   ├── components/               # Data transformation, training, evaluation
│   ├── pipeline/                 # Training and evaluation pipelines
│   ├── entity/                   # Config entity classes
│   └── utils/                    # Utility functions
│
├── params.yaml                   # Model and training hyperparameters
├── schema.yaml                   # Data schema (feature names, labels)
├── main.py                       # Entry point for pipeline execution
└── README.md                     # Project documentation
```

---

## 📈 Workflow Summary

1. **Data Preparation**
   - Load SCADA data from CSV
   - Drop all-zero (non-informative) sensor columns
   - Impute and scale using `KNNImputer` and `StandardScaler`
   - Address class imbalance with `ADASYN`

2. **Pretraining (Autoencoder)**
   - Train an unsupervised autoencoder on normal-only samples
   - Extract latent codes for each input sample

3. **Transfer Learning (Classifier)**
   - Freeze the encoder
   - Train a classification head using balanced training data
   - Save model, architecture, and training metrics

4. **Evaluation**
   - Use held-out validation data
   - Log precision, recall, F1-score, and AUC
   - Save confusion matrix and SHAP importance plots

---

## 📊 Results

| Metric      | Test Set Value |
|-------------|----------------|
| Accuracy    | ~80%           |
| F1 Score    | ~85%           |
| AUC Score   | ~97% (train) / ~98% (val)* |
| Precision   | ~94%           |
| Recall      | ~91%           |

> *_Note: The AUC anomaly during validation suggests a threshold mismatch or class imbalance during slicing. Use `thresholds=` and match data dimensions when evaluating._*

---

## 📦 Installation

```bash
git clone https://github.com/Shahriyar-1988/WindTurbine_fault_detection.git
cd WindTurbine_fault_detection
pip install -r requirements.txt
```

Ensure the following key dependencies are installed:

- TensorFlow >= 2.12
- scikit-learn
- imbalanced-learn
- matplotlib
- pandas
- mlflow

---

## 🧠 SHAP Analysis

Feature importance was computed using **SHAP values** on the classifier’s predictions, offering insights into which SCADA sensor readings most influenced fault predictions.

---

## 📌 Future Improvements

- Incorporate real-time anomaly alerting
- Extend to multi-class fault detection
- Optimize SHAP for batch-wise inference
- Consider interpretable models like LIME or attention-based networks

---

## 🙏 Acknowledgments

This project was conducted as part of an academic collaboration with the **Renewable Energy Research Group**, supervised by **Dr. Ahmaid**. Special thanks to the SCADA data providers and the MLflow community.

---

## 📬 Contact

**Shahriyar**  
GitHub: [@Shahriyar-1988](https://github.com/Shahriyar-1988)

---

## 📝 License

This repository is open-source and available under the MIT License.