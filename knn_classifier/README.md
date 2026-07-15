# Glass Type Classification using K-Nearest Neighbors (KNN)

A machine learning project that predicts the type of glass using the K-Nearest Neighbors (KNN) classification algorithm. The project includes data preprocessing, feature scaling, model training, hyperparameter tuning using the elbow method, evaluation, and model serialization.

## Features
- Exploratory Data Analysis (EDA)
- Correlation Heatmap
- Pairplot Visualization
- Feature Scaling using StandardScaler
- Train-Test Split
- KNN Classification
- Elbow Method for Optimal K Selection
- Model Evaluation
- Model Serialization using Joblib

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook:

```bash
jupyter notebook KNN_Classifier.ipynb
```

## Machine Learning Workflow
1. Load the Glass dataset.
2. Perform exploratory data analysis.
3. Standardize the features.
4. Split the dataset into training and testing sets.
5. Train a KNN classifier.
6. Evaluate the model using Accuracy, Confusion Matrix, and Classification Report.
7. Use the Elbow Method to determine the optimal value of K.
8. Save the trained model and scaler using Joblib.

## Saved Files
- KNN_Classifier.pkl
- scaler.pkl

## Future Improvements
- Cross-validation
- GridSearchCV for hyperparameter tuning
- Streamlit deployment
- Compare with other classification algorithms

## Author
Basiru Mbye
