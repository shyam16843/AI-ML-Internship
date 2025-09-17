# Customer Segmentation using K-Means Clustering

## Project Description
This project segments customers of an online retail business using K-Means clustering. It includes detailed data preprocessing, feature engineering with RFM analysis, cluster evaluation, visualization, and business interpretation to uncover actionable customer groups.

## 1. Project Objective
Build a meaningful and actionable customer segmentation model to support targeted marketing and retention strategies. The objective is to identify distinct customer segments based on purchasing behaviors and value metrics.

## 2. Dataset Information
- **Source**: Online Retail dataset (`Online Retail.xlsx`), containing UK transactions from a retail store.
- **Records**: Over 500,000 transaction records.
- **Features**: Includes invoice details, product codes, descriptions, prices, quantities, customer IDs, countries, and purchase timestamps.
- **Target**: Unsupervised clustering labels (customer segments).

## 3. Methodology
- **Data Cleaning**: Remove canceled orders, handle missing values, exclude negative quantity or price records.
- **Feature Engineering**:
  - RFM features: Recency, Frequency, Monetary value.
  - Additional: Unique products, Total quantity, Customer lifetime, Average spend per day.
- **Outlier Removal**: Using z-score threshold to remove extreme customers.
- **Scaling**: StandardScaler used for feature normalization.
- **Clustering**: K-Means applied with optimal k chosen by Elbow and Silhouette methods.
- **Visualization**: Multi-faceted exploratory plots including pairplots, PCA, boxplots, and cluster-wise feature distributions.
- **Country-wise Analysis**: Distribution of clusters over top countries to analyze geographic patterns.

## 4. Model Performance and Analysis
- Elbow and Silhouette analysis support selection of 3 optimal clusters.
- Cluster sizes are balanced and clearly distinct.
- Business insights: identifying "Champions", "Loyal Customers", "At Risk", and "Lost Customers" segments based on cluster profiles.

## 5. Visualization Overview
A comprehensive set of visualizations supporting this project is provided separately in the [Visualization Document](visualization.md). This document includes detailed descriptions and analyses of all key plots

### Accessing Visualizations

The actual plot images referenced in the visualization document are stored in the `/images` directory within the project repository.

We recommend reviewing the visualization document alongside the main README for a thorough understanding of the model's performance and insightful data interpretations.

## 6. Business and Research Implications
- Helps tailor marketing efforts by segment.
- Supports customer retention strategies via focused campaign targeting.
- Provides insight into purchase behaviors across geography.

## 7. Project Setup and Requirements

### Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

### Installation
Install dependencies by running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Running the Project
1. Place the `Online Retail.xlsx` dataset in the project directory.
2. Run the main script:
```bash
python customer.py
```

3. The script performs all ETL, clustering, visualizations, and outputs results and reports.
4. Outputs saved:
   - `online_retail_customers_clustered.csv`
   - `online_retail_cluster_centers.csv`

## 8. Future Work
- Implement DBSCAN or alternative clustering for noise detection.
- Develop interactive dashboards for easier business use.
- Explore additional temporal or product-related features.
- Add interpretability tooling such as SHAP.

## 9. Contact
For questions or collaboration:
- **Name**: Ghanashyam T V
- **Email**: ghanashyamtv16@gmail.com
- **LinkedIn**: [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for reviewing the Customer Segmentation project. Feel free to reach out for questions or collaborations.
