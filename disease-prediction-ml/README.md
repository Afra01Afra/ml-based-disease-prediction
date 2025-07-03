# Disease Prediction using Machine Learning

## Project Overview

This is an end-to-end machine learning project that predicts diseases based on user-reported symptoms. The entire code is contained in a single Python file, making it simple and self-contained. It includes data preprocessing, exploratory analysis, handling class imbalance, training multiple machine learning models, evaluation using various metrics, robustness testing, and a prediction function for real-time symptom-based disease prediction.

## What This Project Does

1. **Loads and cleans the data**: Reads `Training.csv` and `Testing.csv` files and removes unnecessary columns.
2. **Explores the data**: Visualizes disease distribution and symptom frequency to understand patterns.
3. **Prepares the data**: Encodes the disease labels and handles class imbalance using SMOTE.
4. **Splits the data**: Divides it into training and validation sets for evaluation.
5. **Trains ML models**: Trains XGBoost, CatBoost, and LightGBM classifiers.
6. **Evaluates models**: Reports accuracy, classification report, confusion matrix, and performs 5-fold cross-validation.
7. **Tests robustness**: Adds noise to the test set to check model reliability.
8. **Compares models**: Identifies the best-performing model based on test accuracy.
9. **Predicts disease from symptoms**: Accepts user-defined symptoms and returns the top 3 probable diseases.

## Technologies Used

* Python 3
* pandas, numpy (data processing)
* seaborn, matplotlib (visualization)
* scikit-learn (model evaluation & preprocessing)
* xgboost, catboost, lightgbm (machine learning models)
* imbalanced-learn (SMOTE for handling class imbalance)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/disease-prediction-ml.git
cd disease-prediction-ml
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

This will execute the entire pipeline from data loading to final disease prediction.

## Example Prediction Usage

```python
example_symptoms = {
    'itching': 1,
    'skin_rash': 1,
    'nodal_skin_eruptions': 1,
    'dischromic_patches': 0
}

result = predict_disease(example_symptoms)
print("Predicted Disease:", result["predicted_disease"])
print("Confidence:", result["confidence"])
print("Top 3 Predictions:")
for disease, prob in result["top3_predictions"]:
    print(f"- {disease}: {prob:.2%}")
```

## requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
lightgbm
imbalanced-learn
jupyter
```

## How the Code is Structured (In One File)

The file is divided into the following clear sections:

* Importing required libraries
* Loading and cleaning the dataset
* Data visualization (EDA)
* Preprocessing and encoding
* SMOTE for balancing
* Model training
* Evaluation on validation and test sets
* Robustness testing with noise
* Final comparison and best model selection
* Disease prediction function using symptoms

## Metrics Used for Evaluation

* Accuracy
* Classification Report
* Confusion Matrix
* Cross-validation (5-Fold)
* Accuracy under noise

## License

This project is licensed under the MIT License.

## Contribution

You are welcome to contribute by submitting issues or pull requests to enhance model performance, add features, or improve code structure.

---

**Note:** This project uses structured datasets. Real-world deployment should include validations, error handling, and clinical supervision.
