# Customer Segmentation - Visualizations

This document provides detailed descriptions and insights into all visualizations created during the Customer Segmentation project.

---

## 1. Elbow Method Plot

![Elbow Method](images/Figure_1.png)

**Description:**  
Line plot showing Within-Cluster-Sum-of-Squares (WCSS) for various k values to identify optimal cluster count.

**Insights:**  
- Elbow point suggests k=3 offers a good trade-off between model complexity and cluster fit.

---

## 2. Silhouette Scores Plot

![Silhouette Scores](images/Figure_2.png)

**Description:**  
Line plot of silhouette scores across different cluster counts, measuring cluster cohesion and separation.

**Insights:**  
- Highest score near k=3 confirms optimal cluster choice.

---

## 3. Customer Distribution Across Clusters

![Customer Distribution](images/Figure_3.png)

**Description:**  
Bar plot showing number of customers in each cluster after K-Means clustering.

**Insights:**  
- Balanced cluster sizes suitable for targeted marketing.

---

## 4. Cluster-wise Feature Relationships

![Cluster Feature Relations](images/Figure_4.png)

**Description:**  
Grid of scatter plots depicting relationships between Recency, Frequency, Monetary values, Customer Lifetime, and Average Spend per Day, colored by cluster.

**Insights:**  
- Visually distinct grouping supports cluster validity.

---

## 5. Boxplot of Cluster Characteristics

![Boxplot Cluster Characteristics](images/Figure_5.png)

**Description:**  
Boxplots for key features across clusters, illustrating distribution and variability.

**Insights:**  
- Highlights feature differences and customer segment profiles.

---

## 6. K-Means Clustering PCA Visualization

![K-Means PCA](images/Figure_6.png)

**Description:**  
Scatter plot of PCA-reduced features colored by cluster, with cluster centers marked.

**Insights:**  
- Clusters are well-separated in lower dimensional space, validating segmentation.

---

## 7. DBSCAN Clustering Visualization

![DBSCAN Clustering](images/Figure_7.png)

**Description:**  
Scatter plot of DBSCAN clusters on a subset of scaled data, identifying core points and noise (-1 label).

**Insights:**  
- Identifies noise customers and different cluster shapes compared to K-Means.

---

## 8. Boxplot of Monetary Values (Outliers)

![Boxplot Monetary](images/Figure_8.png)

**Description:**  
Boxplot showing distribution and outliers for monetary values.

**Insights:**  
- Highlights skew and necessity of log transformation.

---

## 9. RFM Feature Distributions

![RFM Distributions](images/Figure_9.png)

**Description:**  
Histograms of Recency, Frequency, and Monetary features displaying the data distribution.

**Insights:**  
- Shows skewness and spread of customer metrics.

---

## 10. Monetary Value Original vs Log-Transformed

![Original vs Log-Transformed Monetary](images/Figure_10.png)

**Description:**  
Side-by-side histograms comparing raw and log-transformed Monetary values.

**Insights:**  
- Log transform reduces right skew, normalizing data for clustering.

---

## 11. Pairplot of RFM Features

![RFM Pairplot](images/Figure_11.png)

**Description:**  
Pairplot visualizing pairwise relationships among RFM features, colored by cluster.

**Insights:**  
- Reveals how clusters separate across feature combinations.

---

## 12. Cluster Distribution by Top 5 Countries

![Cluster by Country](images/Figure_12.png)

**Description:**  
Stacked bar chart showing cluster percentages within top 5 customer countries.

**Insights:**  
- Geographic variation in cluster composition.

---

*Note:* All images are stored in the `/images/` directory.

---

