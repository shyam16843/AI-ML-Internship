# Customer Segmentation using K-Means Clustering

## Project Description
This project segments customers of an online retail business using K-Means clustering. It includes detailed data preprocessing, feature engineering with RFM analysis, cluster evaluation, visualization, and business interpretation to uncover actionable customer groups.

## 1. Project Objective
Develop a machine learning pipeline to segment retail customers based on purchasing behavior and value metrics. This helps in understanding customer groups and tailoring marketing approaches for improved retention and revenue growth.


## 2. Dataset Information
- **Source**: Online Retail dataset (`Online Retail.xlsx`), containing UK transactions from a retail store.
- **Records**: 541,909 transaction records from an online retailer.
- 4,338 unique customers after preprocessing.
- **Features**:include customers' purchase frequency, monetary spend, product diversity, and lifetime metrics.
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

## 4. Cluster Characteristics & Interpretation
| Cluster | Size (No. Customers) | Recency (Days) | Frequency (Purchases) | Monetary ($) | Avg. Spend / Day ($) | Lifetime (Days) | Product Diversity |
|---|---|---|---|---|---|---|---|
| 0 | 1,539 | 58.6 | 3.2 | 914.35 | 12.7 | 170.6 | 151 |
| 1 | 641 | 35.2 | 8.3 | 3,631.91 | 65.3 | 259.1 | 103 |
| 2 | 86 | 153.5 | 1.3 | 11353.87 | 180.5 | 15.9 | 22 |

### Insights
- **Cluster 0:** Moderate activity and spending, representing steady customers.
- **Cluster 1:** High frequency and spending, loyal customers with diverse product interests.
- **Cluster 2:** Low frequency, high spenders but short customer lifetime — potentially new or one-time big buyers.

## 5. Visualization Overview
A comprehensive set of visualizations supporting this project is provided separately in the [Visualization Document](Visualizations.md). This document includes detailed descriptions and analyses of all key plots

### Accessing Visualizations

The actual plot images referenced in the visualization document are stored in the `/images` directory within the project repository.

We recommend reviewing the visualization document alongside the main README for a thorough understanding of the model's performance and insightful data interpretations.

## 6. Business Recommendations
- **Cluster 0:** Implement promotions to encourage higher purchase frequency.
- **Cluster 1:** Target with loyalty programs and exclusive offers.
- **Cluster 2:** Drive re-engagement with personalized outreach campaigns.
- Monitor migration between clusters for dynamic targeting.

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
