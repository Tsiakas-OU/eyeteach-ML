import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load datasets
df = pd.read_csv('../datasets/mecoL1/MECO-en_uk-passage.csv')
flesch_df = pd.read_csv('../datasets/mecoL1/flesch_scores_eng_MECO_L1.csv', 
                       usecols=['trialid', 'flesch_readability'])

# Create dictionary and map flesch_readability
flesch_dict = flesch_df.set_index('trialid')['flesch_readability'].astype(float).to_dict()
df['flesch_readability'] = df['trialid'].map(flesch_dict)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Define all features
features = ['trial.nwords', 'nblink', 'nrun', 'nfix', 'nout', 
           'sac', 'skip', 'refix', 'reg', 'mfix', 
           'firstpass', 'rereading', 'total', 'rate', 'flesch_readability', 'ACCURACY']

# Basic statistics
print("\nBasic Statistics:")
print(df[features].describe())

# Correlation matrix
correlation_matrix = df[features].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Strong correlations
corr_pairs = correlation_matrix.unstack().sort_values(key=abs, ascending=False)
corr_pairs = corr_pairs[corr_pairs != 1.0]
strong_correlations = corr_pairs[abs(corr_pairs) > 0.5]
print("\nStrong correlations (|r| > 0.5):")
print(strong_correlations.head(10))

print("\nCLUSTERING")
nclusters = 3
# Prepare and scale data
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means 
kmeans = KMeans(n_clusters=nclusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df_clustered = X.copy()
df_clustered['cluster'] = cluster_labels

print(f"Cluster distribution:")
cluster_counts = df_clustered['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"  Cluster {cluster}: {count} samples")

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f"PC1 variance: {explained_variance[0]:.3f}")
print(f"PC2 variance: {explained_variance[1]:.3f}")

# Visualize clusters in 2D PCA space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('2D PCA Visualization with Clusters')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Add cluster centers in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, 
           label='Cluster Centers', edgecolors='black')
plt.legend()
plt.tight_layout()
plt.show()


print("\nFEATURE - PCA COMPONENT RELATIONSHIPS")
# Get PCA component weights
components = pca.components_
loadings_df = pd.DataFrame(components.T, 
                          columns=[f'PC{i+1}' for i in range(len(features))],
                          index=features)

print("PCA Component Loadings (First 2 Components):")
print(loadings_df[['PC1', 'PC2']].round(3))

# Visualize feature contributions to first two components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# PC1 feature contributions
sorted_idx_pc1 = np.argsort(np.abs(components[0, :]))[::-1]
features_sorted_pc1 = [features[i] for i in sorted_idx_pc1]
values_pc1 = components[0, sorted_idx_pc1]

ax1.barh(features_sorted_pc1[:8], values_pc1[:8])
ax1.set_xlabel('Loading Value')
ax1.set_title('Top Features Contributing to PC1')
ax1.grid(True, alpha=0.3)

# PC2 feature contributions
sorted_idx_pc2 = np.argsort(np.abs(components[1, :]))[::-1]
features_sorted_pc2 = [features[i] for i in sorted_idx_pc2]
values_pc2 = components[1, sorted_idx_pc2]

ax2.barh(features_sorted_pc2[:8], values_pc2[:8])
ax2.set_xlabel('Loading Value')
ax2.set_title('Top Features Contributing to PC2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nCLUSTER CHARACTERISTICS")

# Analyze feature means by cluster
cluster_means = df_clustered.groupby('cluster').mean()
print("Cluster means:")
print(cluster_means.round(3))

# Visualize cluster characteristics
plt.figure(figsize=(12, 8))
cluster_means_standardized = (cluster_means - cluster_means.mean()) / cluster_means.std()
sns.heatmap(cluster_means_standardized.T, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Standardized Feature Means by Cluster')
plt.tight_layout()
plt.show()

# Feature variation across clusters
feature_variation = cluster_means.std() / cluster_means.mean()
top_differentiating = feature_variation.sort_values(ascending=False).head(8)
print("\nTop features differentiating clusters:")
for feature, variation in top_differentiating.items():
    print(f"  {feature}: {variation:.3f}")
