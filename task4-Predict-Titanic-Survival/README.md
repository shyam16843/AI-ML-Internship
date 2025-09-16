# Titanic Survival Prediction

## Project Description
This project predicts the survival of Titanic passengers using classic machine learning techniques. It includes comprehensive data preprocessing, exploratory data analysis, feature engineering, model training with hyperparameter tuning, and thorough evaluation. Business insights are derived from the model to understand key survival factors.

## 1. Project Objective
The goal is to build an accurate predictive model for Titanic passenger survival, leveraging passenger demographics, travel details, and engineered features. The project demonstrates the end-to-end ML workflow from raw data to deployment-ready prediction.

## 2. Dataset Information
- **Source**: Titanic passenger dataset (`titanic.csv`) from [Data Science Dojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Records**: 891 passenger entries
- **Features** include demographic and ticket information like Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, and engineered features such as FamilySize, IsAlone, AgeGroup, FarePerPerson, and Title.
- **Target Variable**: `Survived` (0 = Did not survive, 1 = Survived)

## 3. Methodology
- **Preprocessing**: 
  - Missing value imputation for Age (median per Pclass, Sex group) and Embarked (most frequent)
  - Dropped high-missing-value Cabin column
  - Encoding categorical variables (Sex and Embarked) into numeric features
- **Feature Engineering**:
  - Created FamilySize, IsAlone, FarePerPerson, AgeGroup categories, and extracted Titles from passenger names
- **Scaling**: Standard scaling applied to numerical features (Age, Fare, FarePerPerson, etc.)
- **Model Training**: Logistic Regression and Random Forest
- **Hyperparameter Tuning**: GridSearchCV optimizing logistic regression parameters on penalties, solvers, and regularization
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC AUC, cross-validation, confusion matrix, and detailed classification reports

## 4. Model Performance
| Model               | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 84.92%   | 83.87%    | 75.36% | 79.39%   | 87.94%  |
| Random Forest       | 81.01%   | 77.78%    | 71.01% | 74.24%   | 82.68%  |

## 5. Key Insights from the Model
### Most Important Survival Factors:
- Being female (`Sex`)
- Titles such as Master and Mrs. (`Title_Master, Title_Mrs`)
- Passenger class (`Pclass`)
- Family size (`FamilySize`)

### Model Confidence Analysis:
- High confidence predictions: ~42 samples (23%)
- Low confidence predictions: ~100 samples (56%)
- Uncertain predictions: ~37 samples (21%)

### Demographic Patterns Confirmed:
- Women and children had significantly higher survival rates
- Higher-class passengers had better chances of survival

## 6. Visualization Overview

A comprehensive set of visualizations supporting this project is provided separately in the [Visualization Document](visualization.md). This document includes detailed descriptions and analyses of all key plots such as:

- Feature importance charts for both Logistic Regression and Random Forest models.
- Confusion matrices demonstrating classification strengths and error patterns.
- ROC curves assessing model discrimination power.
- Prediction examples aligned with real-world passenger profiles.

### Accessing Visualizations

The actual plot images referenced in the visualization document are stored in the `/images` directory within the project repository.

We recommend reviewing the visualization document alongside the main README for a thorough understanding of the model's performance and insightful data interpretations.


## 7. Business and Research Implications
- The model offers insight into historical survival factors.
- Useful for educational purposes and demonstrating ML pipelines.
- Highlights importance of demographics and social context in survival analysis.

## 8. Technical Implementation Insights
- Comprehensive feature engineering from raw dataset
- Effective missing data handling and encoding strategies
- Use of hyperparameter tuning to optimize model
- Visualization for interpretability (feature importance, ROC, confusion matrices)

## 9. Project Setup and Requirements
### Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

### Installation
Install dependencies by running:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the Project
1. Ensure the `titanic.csv` dataset is downloaded or accessible in project root.
2. Run the main script:

```bash
python titanic.py
```

3. The script will perform data loading, preprocessing, model training, evaluation, and generate prediction examples.
4. It outputs an extensive report (`titanic_model_report.txt`) summarizing results and business insights.

## 10. Project Deliverables
- Complete, reproducible Python code for Titanic survival prediction
- Model artifacts including saved model and scaler
- Detailed evaluation metrics and visualizations
- Business insights report
- Sample prediction demonstration

## 11. Future Work
- Incorporate advanced models (e.g., gradient boosting, neural networks)
- Deploy model with APIs or web interface
- Explore temporal survival patterns or additional feature engineering
- Add model interpretability tools like SHAP or LIME for deeper insights

## 12. Contact
For questions or collaboration:
- **Name**: Ghanashyam T V
- **Email**: ghanashyamtv16@gmail.com
- **LinkedIn**: [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring this Titanic survival prediction project. Please feel free to reach out for questions or contribute improvements.

