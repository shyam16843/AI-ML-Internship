# Customer Segmentation using K-Means Clustering
# For Internship Project - Online Retail Dataset

## Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('fivethirtyeight')

## Load and Explore the Online Retail Dataset
# Load the dataset
df = pd.read_excel('Online Retail.xlsx')

# Explore the dataset
print("Dataset Overview:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check unique values
print("\nUnique Values:")
print(f"Customers: {df['CustomerID'].nunique()}")
print(f"Products: {df['StockCode'].nunique()}")
print(f"Countries: {df['Country'].nunique()}")
print(f"Invoices: {df['InvoiceNo'].nunique()}")

# -------------------------------
# Data Cleaning and Preprocessing
# -------------------------------
print(f"Original shape: {df.shape}")
df_clean = df.dropna(subset=['CustomerID'])
print(f"After removing rows with missing CustomerID: {df_clean.shape}")

# Remove canceled orders
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]

# Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# Calculate total amount
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Remove negative quantities and prices
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['UnitPrice'] > 0]

print(f"Final cleaned dataset shape: {df_clean.shape}")
print(f"Final unique customers: {df_clean['CustomerID'].nunique()}")

## RFM Analysis (Recency, Frequency, Monetary)
# Set reference date (one day after the last invoice date)
reference_date = df_clean['InvoiceDate'].max() + timedelta(days=1)

# Calculate RFM values for each customer
rfm_df = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalAmount': 'sum'  # Monetary
}).reset_index()

# Rename columns
rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("RFM DataFrame Overview:")
print(rfm_df.head())
print(f"\nRFM DataFrame Shape: {rfm_df.shape}")

# Log transformation for Monetary to handle outliers
rfm_df['MonetaryLog'] = np.log1p(rfm_df['Monetary'])

## Additional Features for Clustering
# Calculate additional customer metrics
customer_metrics = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': ['min', 'max'],  # First purchase, last purchase
    'StockCode': 'nunique',  # Unique products purchased
    'Quantity': ['sum', 'mean'],  # Total quantity, average quantity per transaction
}).reset_index()

# Flatten column names
customer_metrics.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase', 
                           'UniqueProducts', 'TotalQuantity', 'AvgQuantity']

# Calculate customer lifetime in days
customer_metrics['CustomerLifetime'] = (customer_metrics['LastPurchase'] - customer_metrics['FirstPurchase']).dt.days

# Calculate average spending per day
customer_metrics['AvgSpendPerDay'] = customer_metrics['TotalQuantity'] / customer_metrics['CustomerLifetime'].replace(0, 1)

# Merge with RFM data
customer_features = pd.merge(rfm_df, customer_metrics, on='CustomerID')

# Select final features for clustering
features_for_clustering = ['Recency', 'Frequency', 'MonetaryLog', 'UniqueProducts', 
                          'TotalQuantity', 'CustomerLifetime', 'AvgSpendPerDay']

X = customer_features.set_index('CustomerID')[features_for_clustering]

print("Final Features for Clustering:")
print(X.head())
print(f"\nShape: {X.shape}")

# -------------------------------
# Outlier Removal
# -------------------------------
z_scores = np.abs(stats.zscore(X))
filter_mask = (z_scores < 3).all(axis=1)
X_filtered = X[filter_mask].copy()

print(f"Original samples: {len(X)}")
print(f"After outlier removal: {len(X_filtered)}")
print(f"Outliers removed: {len(X) - len(X_filtered)}")

# -------------------------------
# Feature Scaling (on filtered data)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# -------------------------------
# Determining Optimal Number of Clusters
# -------------------------------
wcss = []
silhouette_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    if k > 1:
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Add a main title for the entire figure
fig.suptitle('Determining Optimal Number of Clusters', fontsize=16, fontweight='bold', y=0.98)

# Elbow method plot
ax1.plot(k_range, wcss, 'bo-', label='WCSS')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WCSS')
ax1.set_title('Elbow Method')
ax1.set_xticks(k_range)
ax1.grid(True)
ax1.legend()

# Silhouette scores plot
ax2.plot(range(2, 8), silhouette_scores, 'go-', label='Silhouette')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Scores')
ax2.set_xticks(range(2, 8))
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

print("\nSilhouette Scores:")
for k, score in zip(range(2, 8), silhouette_scores):
    print(f"k={k}: Silhouette={score:.3f}")

optimal_k = 3  # Choose based on elbow + silhouette
print(f"\nSelected optimal k: {optimal_k}")

# -------------------------------
# K-Means Clustering
# -------------------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
kmeans.fit(X_scaled)

# Create customer_clusters from the filtered data
customer_clusters = customer_features[customer_features['CustomerID'].isin(X_filtered.index)].copy()
customer_clusters.reset_index(drop=True, inplace=True)
customer_clusters['Cluster'] = kmeans.labels_

print("Cluster sizes:")
cluster_counts = customer_clusters['Cluster'].value_counts().sort_index()
print(cluster_counts)

# Plot cluster distribution
plt.figure(figsize=(8,5))
ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')

plt.title('Customer Distribution Across Clusters')
plt.xlabel('Cluster Label')  # Adjusted x-axis title
plt.ylabel('Number of Customers')

# Move subplot contents to the left
plt.subplots_adjust(left=0.15, right=0.85)  # left edge more inward, right edge slightly inward

# Add value labels on top of the bars
for i, count in enumerate(cluster_counts.values):
    ax.text(i, count + 10, str(count), ha='center', va='bottom')
# Correct way to create handles for the legend
# Build one legend entry (patch) for each cluster
handles = [
    mpatches.Patch(color=ax.patches[i].get_facecolor(), label=f'Cluster {cluster_counts.index[i]}')
    for i in range(len(cluster_counts))
]
plt.legend(handles=handles, title="Clusters", loc='upper right', bbox_to_anchor=(1.21,1.1))

plt.show()


# Compute cluster centers in original scale
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers_original, columns=features_for_clustering)
cluster_centers_df['Cluster'] = range(optimal_k)

print("\nCluster Centers (Original Scale):")
print(cluster_centers_df.round(2))

# -------------------------------
# Visualization
# -------------------------------
# Merge cluster labels back to original features for visualization
customer_features_with_clusters = customer_features.copy()
customer_features_with_clusters['Cluster'] = np.nan

# Create a mapping from CustomerID to cluster label
cluster_mapping = dict(zip(X_filtered.index, kmeans.labels_))

# Assign clusters using the mapping
customer_features_with_clusters['Cluster'] = customer_features_with_clusters['CustomerID'].map(cluster_mapping)

# RFM scatter plots
rfm_features = ['Recency', 'Frequency', 'MonetaryLog']
plot_df = customer_features_with_clusters[rfm_features + ['Cluster']].copy()
plot_df['Monetary'] = np.expm1(plot_df['MonetaryLog'])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Recency vs Frequency
scatter1 = axes[0, 0].scatter(plot_df['Recency'], plot_df['Frequency'], 
                             c=plot_df['Cluster'], cmap='viridis', alpha=0.7, s=40)
axes[0, 0].set_xlabel('Recency (Days since last purchase)', fontsize=12)
axes[0, 0].set_ylabel('Frequency (Number of purchases)', fontsize=12)
axes[0, 0].set_title('Recency vs Frequency', fontsize=12)
plt.colorbar(scatter1, ax=axes[0, 0])

# Plot 2: Frequency vs Monetary Value
scatter2 = axes[0, 1].scatter(plot_df['Frequency'], plot_df['Monetary'], 
                             c=plot_df['Cluster'], cmap='viridis', alpha=0.7, s=40)
axes[0, 1].set_xlabel('Frequency (Number of purchases)', fontsize=12)
axes[0, 1].set_ylabel('Monetary Value ($)', fontsize=12)
axes[0, 1].set_title('Frequency vs Monetary Value', fontsize=12)
axes[0, 1].ticklabel_format(style='plain', axis='y')
plt.colorbar(scatter2, ax=axes[0, 1])

# Plot 3: Recency vs Monetary Value
scatter3 = axes[1, 0].scatter(plot_df['Recency'], plot_df['Monetary'], 
                             c=plot_df['Cluster'], cmap='viridis', alpha=0.7, s=40)
axes[1, 0].set_xlabel('Recency (Days since last purchase)', fontsize=12)
axes[1, 0].set_ylabel('Monetary Value ($)', fontsize=12)
axes[1, 0].set_title('Recency vs Monetary Value', fontsize=12)
axes[1, 0].ticklabel_format(style='plain', axis='y')
plt.colorbar(scatter3, ax=axes[1, 0])

# Plot 4: Customer Lifetime vs Avg Spend Per Day
scatter4 = axes[1, 1].scatter(customer_features_with_clusters['CustomerLifetime'], 
                             customer_features_with_clusters['AvgSpendPerDay'], 
                             c=customer_features_with_clusters['Cluster'], cmap='viridis', alpha=0.7, s=40)
axes[1, 1].set_xlabel('Customer Lifetime (Days)', fontsize=12)
axes[1, 1].set_ylabel('Average Spend Per Day ($)', fontsize=12)
axes[1, 1].set_title('Customer Lifetime vs Average Daily Spend', fontsize=12)
axes[1, 1].ticklabel_format(style='plain', axis='y')
plt.colorbar(scatter4, ax=axes[1, 1])

# Add a suptitle above all plots
plt.suptitle('Cluster-wise Feature Relationships', fontsize=24, y=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Create figure-level legend with color patches for each cluster
n_clusters = len(plot_df['Cluster'].unique())
colors = [plt.cm.viridis(i / (n_clusters - 1)) for i in range(n_clusters)]
handles = [mpatches.Patch(color=colors[i], label=f'Cluster {i}') for i in range(n_clusters)]
labels = [f'Cluster {i}' for i in range(n_clusters)]
fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=n_clusters)

plt.show()

# Boxplot visualization
features_to_plot = ['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'CustomerLifetime', 'AvgSpendPerDay']
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Cluster Characteristics by Feature', fontsize=20, y=0.99)

axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    sns.boxplot(x='Cluster', y=feature, data=customer_features_with_clusters, ax=axes[i], palette='viridis')
    axes[i].set_title(f'{feature} by Cluster', fontsize=12)
    axes[i].set_xlabel('Cluster', fontsize=12)
    axes[i].set_ylabel(feature, fontsize=12)
    
    if feature in ['Monetary', 'AvgSpendPerDay']:
        axes[i].ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.show()

# -------------------------------
# Cluster Analysis
# -------------------------------
cluster_summary = customer_features_with_clusters.groupby('Cluster').agg({
    'Recency': ['mean', 'std'],
    'Frequency': ['mean', 'std'],
    'Monetary': ['mean', 'std'],
    'UniqueProducts': ['mean', 'std'],
    'CustomerLifetime': ['mean', 'std'],
    'AvgSpendPerDay': ['mean', 'std']
}).round(2)

print("Cluster Summary Statistics:")
print(cluster_summary)

# Add this code after your existing visualizations
# -------------------------------
# ADDING MISSING PLOTS
# -------------------------------
print("\nGenerating additional visualization plots...")
# 1. K-Means Clustering (PCA Reduced) - More detailed version
# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7, s=50)
plt.title('K-Means Clustering (PCA Reduced)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
# Add cluster centers to the plot
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, alpha=0.8, label='Cluster Centers')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# 2. DBSCAN Clustering (Noise = -1)
# Use a subset of data for better DBSCAN performance
dbscan_sample = X_scaled[:2000]  # Use first 2000 samples
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(dbscan_sample)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(dbscan_sample[:, 0], dbscan_sample[:, 1], c=db_labels, cmap='viridis', alpha=0.7, s=50)
plt.title('DBSCAN Clustering (Noise = -1)')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
# Count and display noise points
noise_count = sum(db_labels == -1)
noise_percentage = (noise_count / len(db_labels)) * 100
plt.figtext(0.15, 0.85, f"Noise points: {noise_count} ({noise_percentage:.1f}%)", 
            bbox=dict(facecolor='white', alpha=0.7))
plt.colorbar(scatter, label='Cluster (Noise = -1)')
plt.grid(True, alpha=0.3)
plt.show()
# 3. Boxplot of Monetary Values (Showing Outliers)
plt.figure(figsize=(10, 6))
sns.boxplot(x=rfm_df['Monetary'], color='lightcoral')
plt.title('Boxplot of Monetary Values (Showing Outliers)')
plt.xlabel('Monetary Value ($)')
plt.ticklabel_format(style='plain', axis='x')
# Add outlier information
Q1 = rfm_df['Monetary'].quantile(0.25)
Q3 = rfm_df['Monetary'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
outliers_count = sum(rfm_df['Monetary'] > outlier_threshold)
outlier_percentage = (outliers_count / len(rfm_df)) * 100
plt.figtext(0.15, 0.8, f"Outliers (> ${outlier_threshold:,.0f}): {outliers_count} customers ({outlier_percentage:.1f}%)", 
            bbox=dict(facecolor='white', alpha=0.7))
plt.show()

print("Additional plots generated successfully!") 

# -------------------------------
# Business Interpretation
# -------------------------------
print("\nCLUSTER INTERPRETATION:")
overall_means = customer_features_with_clusters[customer_features_with_clusters['Cluster'].notna()].mean()

for cluster_id in range(optimal_k):
    cluster_data = customer_features_with_clusters[customer_features_with_clusters['Cluster'] == cluster_id]
    
    print(f"\nCluster {cluster_id} (n={len(cluster_data)} customers):")
    print(f"  - Recency: {cluster_data['Recency'].mean():.1f} days since last purchase")
    print(f"  - Frequency: {cluster_data['Frequency'].mean():.1f} purchases")
    print(f"  - Monetary: ${cluster_data['Monetary'].mean():.2f} total spending")
    print(f"  - Unique Products: {cluster_data['UniqueProducts'].mean():.1f} different items")
    print(f"  - Customer Lifetime: {cluster_data['CustomerLifetime'].mean():.1f} days")
    print(f"  - Avg Spend/Day: ${cluster_data['AvgSpendPerDay'].mean():.2f}")

# -------------------------------
# Save Results
# -------------------------------
output_df = customer_features_with_clusters.copy()
customer_country = df_clean.groupby('CustomerID')['Country'].first()
output_df = output_df.merge(customer_country, on='CustomerID', how='left')

output_df.to_csv('online_retail_customers_clustered.csv', index=False)
cluster_centers_df.to_csv('online_retail_cluster_centers.csv', index=False)

print("\nResults saved successfully!")
print("Files created:")
print("- online_retail_customers_clustered.csv")
print("- online_retail_cluster_centers.csv")

# -------------------------------
# Advanced Analysis: Country-wise Segmentation
# -------------------------------
if 'Country' in output_df.columns:
    print("\nCOUNTRY-WISE CLUSTER ANALYSIS:")
    print("=" * 30)
    
    # Cluster distribution by country
    country_cluster = pd.crosstab(output_df['Country'], output_df['Cluster'])
    country_cluster_pct = country_cluster.div(country_cluster.sum(axis=1), axis=0) * 100
    
    # Top countries by customer count
    top_countries = output_df['Country'].value_counts().head(5).index
    
    print("\nCluster Distribution in Top 5 Countries (%):")
    for country in top_countries:
        if country in country_cluster_pct.index:
            print(f"\n{country}:")
            for cluster_id in range(optimal_k):
                pct = country_cluster_pct.loc[country, cluster_id]
                print(f"  Cluster {cluster_id}: {pct:.1f}%")
    
    # Visualize cluster distribution by country
    #plt.figure(figsize=(8, 5))
    ax = country_cluster_pct.loc[top_countries].T.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
    plt.title('Cluster Distribution in Top 5 Countries')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage of Customers')
    plt.xticks(rotation=0)
    
    # Adjust plot area to be a bit to the left (more room on right for legend)
    plt.subplots_adjust(right=0.8)

    # Move legend to upper right outside plot
    plt.legend(title='Country', loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()

# -------------------------------
# Additional Visualizations
# -------------------------------

# 1. RFM Distribution Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Distribution of RFM Values', fontsize=20)

# Recency
sns.histplot(rfm_df['Recency'], ax=axes[0], kde=True, bins=30)
axes[0].set_title('Distribution of Recency', fontsize=12)
axes[0].set_xlabel('Recency (Days)', fontsize=12)

# Frequency
sns.histplot(rfm_df['Frequency'], ax=axes[1], kde=True, bins=30)
axes[1].set_title('Distribution of Frequency', fontsize=12)
axes[1].set_xlabel('Frequency', fontsize=12)

# Monetary
sns.histplot(rfm_df['Monetary'], ax=axes[2], kde=True, bins=30)
axes[2].set_title('Distribution of Monetary', fontsize=12)
axes[2].set_xlabel('Monetary Value ($)', fontsize=12)
axes[2].ticklabel_format(style='plain', axis='x')

plt.tight_layout()
plt.show()

# 2. Monetary Value Before and After Log Transformation
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
fig.suptitle('Monetary Value Distribution: Original vs Log-Transformed', fontsize=24)

# Original Monetary
sns.histplot(rfm_df['Monetary'], ax=axes[0], kde=True, bins=30)
axes[0].set_title('Original Monetary Distribution', fontsize=12)
axes[0].set_xlabel('Monetary Value ($)', fontsize=12)
axes[0].ticklabel_format(style='plain', axis='x')

# Log-transformed Monetary
sns.histplot(rfm_df['MonetaryLog'], ax=axes[1], kde=True, bins=30)
axes[1].set_title('Log-Transformed Monetary Distribution', fontsize=12)
axes[1].set_xlabel('Log(Monetary Value)', fontsize=12)

plt.tight_layout()
plt.show()

# 3. Pairplot of RFM features
g = sns.pairplot(plot_df, hue='Cluster', palette='viridis', diag_kind='kde', 
                 plot_kws={'alpha': 0.6, 's': 30}, height=2.5)

# Set axis label font sizes smaller
for ax in g.axes.flatten():
    ax.xaxis.label.set_size(10)  # x-axis label font size
    ax.yaxis.label.set_size(10)  # y-axis label font size
    ax.tick_params(axis='both', labelsize=8)  # tick labels size

plt.suptitle('RFM Analysis: Pairwise Relationships Colored by Cluster', y=0.95, fontsize=20)
plt.subplots_adjust(top=0.9)  
plt.show()

# 4. PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)
plt.title('PCA Visualization of Clusters')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------------
# Business Recommendations
# -------------------------------
print("\nBUSINESS RECOMMENDATIONS:")
print("=" * 30)

for cluster_id in range(optimal_k):
    cluster_data = customer_features_with_clusters[customer_features_with_clusters['Cluster'] == cluster_id]
    
    recency = cluster_data['Recency'].mean()
    frequency = cluster_data['Frequency'].mean()
    monetary = cluster_data['Monetary'].mean()
    
    print(f"\nCluster {cluster_id}:")
    if recency < 30 and frequency > 5:
        print("  - Champions: Offer premium products and exclusive benefits")
    elif recency < 90 and monetary > 500:
        print("  - Loyal Customers: Cross-sell complementary products")
    elif recency < 180:
        print("  - At Risk: Reactivation campaigns with special discounts")
    else:
        print("  - Lost Customers: Win-back campaigns with significant incentives")

print("\nProject completed successfully!")