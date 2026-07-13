# House Price Prediction 
 
This project applies regression techniques to predict California housing prices using features such as location, median income, and housing characteristics. 
 
## ?? Project Structure 
- house_price_prediction2.ipynb - Jupyter notebook with preprocessing, training, and evaluation. 
- requirements.txt - Python dependencies. 
- house_price_model.pkl - Saved trained regression model. 
- independent_scaler.pkl - Saved scaler for preprocessing. 
 
## ?? How to Run 
1. Clone the repository: 
   cd house-price-prediction 
2. Install dependencies: 
   pip install -r requirements.txt 
3. Open the notebook: 
   jupyter notebook house_price_prediction2.ipynb 
4. Load the saved model and scaler: 
   import joblib 
   loaded_model = joblib.load("house_price_Prediction_model.pkl") 
   scaler = joblib.load("independent_scaler.pkl") 
 
## ?? Features 
- longitude 
- latitude 
- housing_median_age 
- total_rooms 
- total_bedrooms 
- population 
- households 
- median_income 
- ocean_proximity (categorical) 
 
## ??? Tech Stack 
- Python 
- Pandas, NumPy 
- Scikit-learn 
- Seaborn, Matplotlib 
- Jupyter Notebook 
 
## ? Future Improvements 
- Hyperparameter tuning 
- Feature engineering 
- Deployment with Flask/Django 
