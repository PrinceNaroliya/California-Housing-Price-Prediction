# California-Housing-Price-Prediction

## Overview
This project predicts the median house value in California using features such as longitude, latitude, number of rooms, households, median income, and ocean proximity. The model is built using **XGBoost Regressor** and demonstrates feature engineering, preprocessing, and hyperparameter tuning for optimal performance.

---

## Dataset
- **Source:** [California Housing Dataset](/kaggle/input/california-house-price-prediction/california_housing_test_1.csv)
- **Note:** Dataset is **not included** in this repository. To run the notebook, download the dataset from Kaggle and place it in the `data/` folder or update the path in the notebook.  

- **Description:** The dataset contains the following features:

| Feature | Description |
|---------|-------------|
| longitude | Longitude coordinate of the house block |
| latitude | Latitude coordinate of the house block |
| housing_median_age | Median age of houses in the block |
| total_rooms | Total number of rooms in the block |
| total_bedrooms | Total number of bedrooms in the block |
| population | Total population in the block |
| households | Total households in the block |
| median_income | Median income of households |
| median_house_value | Target variable (price of the house) |
| ocean_proximity | Categorical variable indicating distance to the ocean |

---

## Data Preprocessing
- Dropped irrelevant columns (`Unnamed: 9`)  
- Filled missing values in categorical columns with **mode**  
- Converted categorical variables using **One-Hot Encoding**  
- Scaled numerical features using **StandardScaler**  

---

## Feature Engineering
- One-hot encoding for `ocean_proximity`  
- Dropped `ocean_proximity_NEAR BAY` column to avoid multicollinearity  

---

## Model
- **Model Used:** XGBoost Regressor
- **Hyperparameters:**
  - `n_estimators = 500`
  - `max_depth = 6`
  - `learning_rate = 0.05`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`
  - `random_state = 42`
- **Train-Test Split:** 70%-30%  
- **Evaluation Metric:** R² score

---

## Results
- **XGBoost R² Score:** 0.77  

As a beginner-level implementation, this score demonstrates a reasonable baseline. While the model performs decently, there is still room for improvement. Further enhancements could include:

- Advanced feature engineering (interaction features, outlier handling)
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Trying ensemble methods like Stacking, Random Forest, or LightGBM
- Cross-validation to ensure more robust performance

This project serves as a solid starting point and provides a foundation for progressively improving prediction accuracy in future iterations.

---

## How to Run
1. Clone the repo:

    git clone https://github.com/<your-username>/California-Housing-Price-Prediction

2. Install required libraries:
   
    pip install -r requirements.txt

4. Download the dataset from Kaggle and update the path in the notebook if needed.
5. Open notebook:
   
    jupyter notebook notebooks/california_housing_model.ipynb

7. Run all cells to train the model and check predictions.

---

## Future Work

- Add advanced feature engineering (interaction features, outlier removal)

- Experiment with Random Forest and Stacked Models

- Optimize hyperparameters using GridSearchCV / RandomizedSearchCV

- Evaluate using Cross-Validation for robust performance

---

## Libraries Used

- pandas

- numpy

- scikit-learn

- xgboost

---

## Author

Prince – Data Science & Machine Learning Enthusiast
