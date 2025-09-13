## 1. Project Overview
This project builds a linear regression model to predict median house prices in California districts based on demographic and housing features. The goal is to create an accurate pricing model that can assist real estate professionals, homeowners, and policymakers in valuation and decision-making.

## 2. Dataset Information
The project uses the California Housing Dataset from scikit-learn, containing:
- **20,640 samples** from California districts
- **8 feature columns** including median income, housing age, average rooms, etc.
- **Target variable**: Median house value (in $100,000s)
- **Key features**: `MedInc` (median income), `AveRooms`, `HouseAge`, `AveOccup`, `Latitude`, `Longitude`

*Note: The dataset is automatically loaded from scikit-learn and requires no separate download.*

## 3. Methodology
- **Data Preprocessing**: Selected top correlated features and applied train-test split (80/20)
- **Modeling**: Multiple Linear Regression algorithm
- **Feature Selection**: Chose top 3 features based on correlation analysis (`MedInc`, `AveRooms`, `HouseAge`)
- **Validation**: RMSE, MAE, and R² metrics with comparison to baseline model
- **Visualizations**: Correlation heatmaps, actual vs predicted plots, residual analysis

## 4. Model Performance
### Key Results:
- **Root Mean Squared Error (RMSE)**: 0.8117
- **Mean Absolute Error (MAE)**: 0.6033
- **R² Score**: 0.4972
- **Adjusted R² Score**: 0.4968

### Comparison with Baseline:
- **Single feature model** (`MedInc` only): RMSE = 0.8421, R² = 0.4589
- **Three feature model**: **3.6% improvement** in RMSE

## 5. Key Insights Uncovered
- **Strongest Predictor**: Median income (`MedInc`) shows highest correlation (0.688) with house prices
- **Feature Relationships**: 
  - Positive: Higher income → Higher house prices
  - Negative: More rooms → Slightly lower prices (possibly due to multicollinearity)
  - Weak positive: Older houses → Slightly higher prices
- **Error Analysis**: Typical prediction error of $81,170, explaining 49.7% of price variance

## 6. Business Implications
- **Real Estate Valuation**: Model can provide quick price estimates for California districts
- **Policy Making**: Income is the strongest determinant of housing prices, highlighting affordability issues
- **Investment Decisions**: Identifies key factors influencing property values in different regions
- **Margin of Error**: $60,330 average error provides realistic expectations for practical use

## 7. Technical Implementation
### Regression Equation:
MedHouseVal = 0.0173 + 0.4448×MedInc - 0.0281×AveRooms + 0.0168×HouseAge


### Feature Coefficients:
- **MedInc**: +0.4448 (strong positive impact)
- **AveRooms**: -0.0281 (slight negative impact)  
- **HouseAge**: +0.0168 (weak positive impact)

## 8. Future Work
- Experiment with additional features (`Latitude`, `Longitude` for geographic patterns)
- Try polynomial features and regularization techniques
- Implement advanced algorithms (Random Forest, Gradient Boosting)
- Add time-series analysis for price trends over time
- Develop interactive web application for real-time predictions

## 9. Visualizations

Detailed plots can be found in the [VISUALIZATION.md](./VISUALIZATION.md) file.

---


## 10. Installation 

### Requirements:

pip install -r requirements.txt
Run the project:

## How to Run This Project

# Option 1: Jupyter Notebook (Exploratory analysis)
jupyter notebook Linear_Regression_Housing_Prices.ipynb

# Option 2: Python script (Production ready)
python prediction.py


---

## 11. Contact

For questions or collaboration, please reach out:

- **Name:** Ghanashyam T V  
- **Email:** [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)  
- **LinkedIn:** [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring this project! Feel free to open issues or pull requests for improvements.
