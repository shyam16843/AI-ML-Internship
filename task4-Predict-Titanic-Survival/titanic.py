import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, precision_score, recall_score, f1_score, 
                             roc_auc_score, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')

def load_and_preprocess_data():
    """
    Load and preprocess the Titanic dataset
    
    Returns:
    - df_final: Preprocessed DataFrame with all features
    - scaler: Fitted StandardScaler object for later use on new data
    """
    # Load dataset from URL or local file if already downloaded
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    if os.path.exists('titanic.csv'):
        df = pd.read_csv('titanic.csv')
    else:
        df = pd.read_csv(url)
        df.to_csv('titanic.csv', index=False)
    
    # Handle missing values in Age by filling with median based on Pclass and Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Handle missing values in Embarked by filling with the most common value
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Drop Cabin column as it has too many missing values
    df = df.drop('Cabin', axis=1)
    
    # Encode categorical variables
    # Convert Sex to binary (0 for male, 1 for female)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Create dummy variables for Embarked column
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    df = df.drop('Embarked', axis=1)
    
    # Feature engineering - create new features that might be predictive
    # FamilySize: Total number of family members onboard
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone: Flag for passengers traveling alone
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # AgeGroup: Categorize passengers into age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Create dummy variables for AgeGroup
    agegroup_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
    df = pd.concat([df, agegroup_dummies], axis=1)
    df = df.drop('AgeGroup', axis=1)
    
    # FarePerPerson: Calculate fare per person for family groups
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # Extract titles from names for enhanced features
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles together
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                      'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    # Standardize titles
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Create dummy variables for Title
    title_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, title_dummies], axis=1)
    
    # Drop unnecessary columns
    df = df.drop(['Title', 'Name'], axis=1)
    df_final = df.drop(['PassengerId', 'Ticket'], axis=1)
    
    # Scale numerical features to have mean=0 and variance=1
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare', 'FarePerPerson', 'SibSp', 'Parch', 'FamilySize']
    numerical_features = [col for col in numerical_features if col in df_final.columns]
    df_final[numerical_features] = scaler.fit_transform(df_final[numerical_features])
    
    return df_final, scaler

def train_models(X, y):
    """
    Train and evaluate multiple models with hyperparameter tuning
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    
    Returns:
    - best_logreg: Best Logistic Regression model from grid search
    - rf_model: Random Forest model
    - X_train, X_test, y_train, y_test: Train/test splits
    """
    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define parameter grid for Logistic Regression hyperparameter tuning
    # Using different solvers based on penalty type compatibility
    param_grid = [
        {
            'penalty': ['l2'],  # L2 regularization
            'solver': ['lbfgs', 'newton-cg'],  # Solvers compatible with L2
            'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
            'max_iter': [1000]  # Maximum iterations
        },
        {
            'penalty': ['l1'],  # L1 regularization
            'solver': ['liblinear'],  # Solver compatible with L1
            'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
            'max_iter': [1000]  # Maximum iterations
        }
    ]
    
    # Perform grid search to find best Logistic Regression parameters
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),  # Base estimator
        param_grid,  # Parameter grid to search
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',  # Evaluation metric
        n_jobs=-1,  # Use all available CPUs
        verbose=0  # No output during training
    )
    
    # Fit the grid search to training data
    grid_search.fit(X_train, y_train)
    best_logreg = grid_search.best_estimator_
    
    # Train a Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return best_logreg, rf_model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name=""):
    """
    Evaluate a model and return comprehensive metrics
    
    Parameters:
    - model: Trained model to evaluate
    - X_test: Test features
    - y_test: Test labels
    - model_name: Name of the model for display purposes
    
    Returns:
    - Dictionary containing evaluation metrics and predictions
    """
    # Generate predictions and prediction probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print evaluation results
    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_feature_importance(model, feature_names, model_name=""):
    """
    Plot feature importance for a model
    
    Parameters:
    - model: Trained model with feature importance or coefficients
    - feature_names: List of feature names
    - model_name: Name of the model for plot title
    """
    # Handle different model types
    if hasattr(model, 'coef_'):  # Logistic Regression
        # Use absolute coefficient values as importance measure
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance['Feature'][:10], importance['Importance'][:10])
        plt.xlabel('Absolute Coefficient Value')
        plt.title(f'Top 10 Features - {model_name}')
        plt.gca().invert_yaxis()  # Display highest importance at top
        plt.tight_layout()
        plt.show()
        
    elif hasattr(model, 'feature_importances_'):  # Random Forest
        # Use built-in feature importance
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance['Feature'][:10], importance['Importance'][:10])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Features - {model_name}')
        plt.gca().invert_yaxis()  # Display highest importance at top
        plt.tight_layout()
        plt.show()
    
    return importance

def plot_roc_curve(y_test, y_pred_proba, model_name=""):
    """
    Plot ROC curve and calculate AUC
    
    Parameters:
    - y_test: True labels
    - y_pred_proba: Predicted probabilities for positive class
    - model_name: Name of the model for plot title
    """
    # Calculate ROC curve values
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return roc_auc

def plot_confusion_matrix(y_test, y_pred, model_name=""):
    """
    Plot confusion matrix
    
    Parameters:
    - y_test: True labels
    - y_pred: Predicted labels
    - model_name: Name of the model for plot title
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Did Not Survive', 'Survived'], 
                yticklabels=['Did Not Survive', 'Survived'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def predict_survival(passenger_data, model, scaler, feature_names):
    """
    Predict survival probability for new passenger data
    
    Parameters:
    - passenger_data: Dictionary with passenger features
    - model: Trained model
    - scaler: Fitted scaler for feature normalization
    - feature_names: List of feature names in the correct order
    
    Returns:
    - Dictionary with prediction results including survival prediction,
      probability, and confidence level
    """
    # Convert input data to DataFrame
    passenger_df = pd.DataFrame([passenger_data])
    
    # Ensure all features are present (add missing features with default value 0)
    for feature in feature_names:
        if feature not in passenger_df.columns:
            passenger_df[feature] = 0
    
    # Reorder columns to match training data feature order
    passenger_df = passenger_df[feature_names]
    
    # Scale the numerical features using the pre-fitted scaler
    numerical_features = ['Age', 'Fare', 'FarePerPerson', 'SibSp', 'Parch', 'FamilySize']
    numerical_features = [col for col in numerical_features if col in passenger_df.columns]
    
    if numerical_features:
        passenger_df[numerical_features] = scaler.transform(passenger_df[numerical_features])
    
    # Make prediction
    prediction = model.predict(passenger_df)[0]
    probability = model.predict_proba(passenger_df)[0][1]
    
    # Determine confidence level based on probability
    if probability > 0.7 or probability < 0.3:
        confidence = 'High'
    else:
        confidence = 'Medium'
    
    return {
        'survival_prediction': 'Survived' if prediction == 1 else 'Did Not Survive',
        'survival_probability': probability,
        'confidence': confidence
    }

def generate_report(model, X_test, y_test, y_pred, y_pred_proba, feature_importance):
    """
    Generate a comprehensive project report
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test labels
    - y_pred: Predicted labels
    - y_pred_proba: Predicted probabilities
    - feature_importance: DataFrame with feature importance
    
    Returns:
    - Formatted report string
    """
    # Calculate additional metrics for the report
    survival_probs = y_pred_proba
    high_survival = (survival_probs > 0.7).sum()
    low_survival = (survival_probs < 0.3).sum()
    uncertain = len(survival_probs) - high_survival - low_survival
    
    # Get top 5 most important features
    top_features = feature_importance.nlargest(5, 'Importance')
    
    # Create comprehensive report
    report = f"""
TITANIC SURVIVAL PREDICTION PROJECT REPORT
{'=' * 50}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROJECT OVERVIEW:
- Dataset: Titanic passenger data (n={len(X_test) + len(y_test)})
- Target variable: Survival (0 = Did not survive, 1 = Survived)
- Best model: {type(model).__name__}
- Final test accuracy: {accuracy_score(y_test, y_pred):.4f}

DATA PREPROCESSING:
- Handled missing values in Age and Embarked
- Encoded categorical variables (Sex, Embarked)
- Created new features: FamilySize, IsAlone, AgeGroup, FarePerPerson, Title
- Scaled numerical features: Age, Fare, FarePerPerson, SibSp, Parch, FamilySize

MODEL PERFORMANCE:
- Accuracy: {accuracy_score(y_test, y_pred):.4f}
- Precision: {precision_score(y_test, y_pred):.4f}
- Recall: {recall_score(y_test, y_pred):.4f}
- F1-Score: {f1_score(y_test, y_pred):.4f}
- ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}

KEY INSIGHTS:
1. Most important predictive factors:
{top_features.to_string()}

2. Model confidence:
- High confidence predictions: {high_survival}/{len(y_test)}
- Low confidence predictions: {low_survival}/{len(y_test)}
- Uncertain predictions: {uncertain}/{len(y_test)}

3. Demographic patterns confirmed:
- Women had significantly higher survival rates
- Children were prioritized for rescue
- Higher-class passengers had better survival chances

RECOMMENDATIONS:
1. The model can predict survival with {accuracy_score(y_test, y_pred)*100:.1f}% accuracy
2. Most reliable for female passengers and children
3. Use for historical analysis and educational purposes

MODEL LIMITATIONS:
- Based on historical data from 1912
- Limited by available passenger information
- Some demographic groups have small sample sizes
"""
    return report

def demonstrate_prediction(model, scaler, feature_names):
    """
    Demonstrate the prediction function with example passenger data
    
    Parameters:
    - model: Trained model
    - scaler: Fitted scaler
    - feature_names: List of feature names
    """
    # Define example passenger data for demonstration
    examples = [
        {
            'description': "First-class female adult passenger",
            'data': {
                'Pclass': 1, 'Sex': 1, 'Age': 28, 'SibSp': 0, 'Parch': 0, 
                'Fare': 100, 'FamilySize': 1, 'IsAlone': 1, 'FarePerPerson': 100,
                'Embarked_C': 1, 'Embarked_Q': 0, 'Embarked_S': 0,
                'AgeGroup_Child': 0, 'AgeGroup_Teen': 0, 'AgeGroup_Adult': 1,
                'AgeGroup_Middle': 0, 'AgeGroup_Senior': 0,
                'Title_Master': 0, 'Title_Miss': 1, 'Title_Mr': 0, 
                'Title_Mrs': 0, 'Title_Rare': 0
            }
        },
        {
            'description': "Third-class male child passenger",
            'data': {
                'Pclass': 3, 'Sex': 0, 'Age': 8, 'SibSp': 3, 'Parch': 2, 
                'Fare': 25, 'FamilySize': 6, 'IsAlone': 0, 'FarePerPerson': 4.17,
                'Embarked_C': 0, 'Embarked_Q': 0, 'Embarked_S': 1,
                'AgeGroup_Child': 1, 'AgeGroup_Teen': 0, 'AgeGroup_Adult': 0,
                'AgeGroup_Middle': 0, 'AgeGroup_Senior': 0,
                'Title_Master': 1, 'Title_Miss': 0, 'Title_Mr': 0, 
                'Title_Mrs': 0, 'Title_Rare': 0
            }
        }
    ]
    
    print("PREDICTION DEMONSTRATION:")
    print("=" * 50)
    
    # Make predictions for each example
    for example in examples:
        result = predict_survival(example['data'], model, scaler, feature_names)
        print(f"\nExample: {example['description']}")
        print(f"Prediction: {result['survival_prediction']}")
        print(f"Probability: {result['survival_probability']:.2%}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 30)

def main():
    """
    Main function to run the complete Titanic survival prediction analysis
    """
    print("Loading and preprocessing data...")
    df_final, scaler = load_and_preprocess_data()
    
    # Split the data into features (X) and target (y)
    X = df_final.drop('Survived', axis=1)
    y = df_final['Survived']
    
    print(f"Data shape: {df_final.shape}")
    print(f"Features: {list(X.columns)}")
    
    print("Training models...")
    best_logreg, rf_model, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # Evaluate Logistic Regression
    print("\n" + "="*50)
    lr_metrics = evaluate_model(best_logreg, X_test, y_test, "Logistic Regression")
    lr_importance = plot_feature_importance(best_logreg, X.columns, "Logistic Regression")
    plot_roc_curve(y_test, lr_metrics['y_pred_proba'], "Logistic Regression")
    plot_confusion_matrix(y_test, lr_metrics['y_pred'], "Logistic Regression")
    
    # Evaluate Random Forest
    print("\n" + "="*50)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_importance = plot_feature_importance(rf_model, X.columns, "Random Forest")
    plot_roc_curve(y_test, rf_metrics['y_pred_proba'], "Random Forest")
    plot_confusion_matrix(y_test, rf_metrics['y_pred'], "Random Forest")
    
    # Select best model based on accuracy
    if lr_metrics['accuracy'] >= rf_metrics['accuracy']:
        final_model = best_logreg
        final_metrics = lr_metrics
        final_importance = lr_importance
        print("Selected Logistic Regression as final model")
    else:
        final_model = rf_model
        final_metrics = rf_metrics
        final_importance = rf_importance
        print("Selected Random Forest as final model")
    """
    # Generate and save report
    report = generate_report(final_model, X_test, y_test, 
                            final_metrics['y_pred'], final_metrics['y_pred_proba'],
                            final_importance)
    print(report)
    
    # Save report to file
    with open('titanic_model_report.txt', 'w') as f:
        f.write(report)
    print("Report saved as 'titanic_model_report.txt'")
    """
    # Save the final model and scaler for future use
    joblib.dump(final_model, 'final_titanic_model.pkl')
    joblib.dump(scaler, 'titanic_scaler.pkl')
    print("Final model and scaler saved successfully!")
    
    # Demonstrate prediction function with example data
    demonstrate_prediction(final_model, scaler, X.columns)
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()