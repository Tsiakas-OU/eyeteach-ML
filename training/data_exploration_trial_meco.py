import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

# =============================================================================
print(f"Total participants: {df['uniform_id'].nunique()}")
print(f"Total trials: {df['trialid'].nunique()}")
print(f"Total observations: {len(df)}")

# Check for missing values
print("\nMissing values per feature:")
missing_values = df[features].isnull().sum()
print(missing_values[missing_values > 0])

# Basic statistics
print("\nBasic statistics:")
print(df[features].describe())

# Create distribution plots for all features
# Split features into two groups of 8
features_group1 = features[:8]   # First 8 features
features_group2 = features[8:]   # Last 8 features

# Figure 1: First 8 features
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
axes1 = axes1.ravel()

for i, feature in enumerate(features_group1):
    axes1[i].hist(df[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes1[i].set_title(f'Distribution of {feature}', fontsize=12)
    axes1[i].set_xlabel(feature)
    axes1[i].set_ylabel('Frequency')
    axes1[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Figure 2: Last 8 features
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
axes2 = axes2.ravel()

for i, feature in enumerate(features_group2):
    axes2[i].hist(df[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes2[i].set_title(f'Distribution of {feature}', fontsize=12)
    axes2[i].set_xlabel(feature)
    axes2[i].set_ylabel('Frequency')
    axes2[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate correlation matrix
correlation_matrix = df[features].corr()

# Create a large correlation heatmap
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
            center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8},
            annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Prepare data for PCA
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by first 5 components:")
for i in range(5):
    print(f"  PC{i+1}: {explained_variance[i]:.3f} ({explained_variance[i]*100:.1f}%)")

# Show feature contributions to first two components
components = pca.components_
print("\nTop features contributing to PC1:")
pc1_contributions = pd.Series(components[0], index=features).abs().sort_values(ascending=False)
for feature, loading in pc1_contributions.head(5).items():
    print(f"  {feature}: {loading:.3f}")

print("\nTop features contributing to PC2:")
pc2_contributions = pd.Series(components[1], index=features).abs().sort_values(ascending=False)
for feature, loading in pc2_contributions.head(5).items():
    print(f"  {feature}: {loading:.3f}")

# Apply K-means clustering on the PCA-reduced data
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)

print(f"\nCluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Original PCA visualization
ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20, color='blue')
ax1.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
ax1.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
ax1.set_title('PCA Visualization (All Trials)')
ax1.grid(True, alpha=0.3)

# Plot 2: PCA with K-means clustering
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, s=30)
ax2.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
ax2.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
ax2.set_title('K-means Clustering on PCA')

# Add cluster centers
centers_pca = kmeans.cluster_centers_[:, :2]
ax2.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, 
           label='Cluster Centers', edgecolors='black')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Add cluster labels to the original data
df_clustered = X.copy()
df_clustered['cluster'] = cluster_labels

# Analyze mean values for each cluster
cluster_means = df_clustered.groupby('cluster').mean()

print("\nCluster characteristics (mean values for features):")
print(cluster_means[features].round(3))

# Calculate correlation between features and cluster assignments
feature_correlations = {}
for feature in features:
    correlation = np.corrcoef(df_clustered['cluster'], df_clustered[feature])[0, 1]
    feature_correlations[feature] = abs(correlation)
# Sort features by correlation strength
sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)

print("Features most correlated with cluster assignments:")
for feature, corr in sorted_features[:5]:
    print(f"  {feature}: {corr:.3f}")

from sklearn.feature_selection import f_classif

# Calculate F-values (variance between clusters / variance within clusters)
f_values, p_values = f_classif(X_scaled, cluster_labels)
feature_importance = pd.DataFrame({
    'feature': features,
    'f_value': f_values,
    'p_value': p_values
}).sort_values('f_value', ascending=False)

print("Feature importance (ANOVA F-value):")
print(feature_importance)  