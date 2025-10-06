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
print(f"Original dataset shape: {df.shape}")

# Get unique participants
participants = df['uniform_id'].unique()

# Define features
features = ['nblink', 'nrun', 'nfix', 
           'sac', 'skip', 'refix', 'reg', 'mfix', 
           'firstpass', 'rereading', 'total', 'rate', 
           'flesch_readability', 'ACCURACY']

print("\nCREATING PARTICIPANT-LEVEL FEATURES")
participant_features = []
for participant in participants:
    participant_data = df[df['uniform_id'] == participant]
    
    # Calculate mean and std for each feature across all trials
    feature_vector = []
    for feature in features:
        feature_vector.extend([
            participant_data[feature].mean(),    # Mean behavior
            #participant_data[feature].std()      # Variability in behavior
        ])
    
    participant_features.append(feature_vector)

# Convert to DataFrame
participant_features_df = pd.DataFrame(participant_features, index=participants)
feature_columns = []
for feature in features:
    feature_columns.append(f'{feature}_mean')
    #feature_columns.append(f'{feature}_std')
participant_features_df.columns = feature_columns

print(f"Created participant feature matrix: {participant_features_df.shape}")

# Check for missing values and remove participants with missing data
missing_values = participant_features_df.isnull().sum().sum()
print(f"Total missing values: {missing_values}")

if missing_values > 0:
    participant_features_df = participant_features_df.dropna()
    print(f"Participants after removing missing values: {len(participant_features_df)}")

# =============================================================================

print("\n1. BASIC STATISTICS AND CORRELATIONS")

# Basic statistics
print("\nParticipant-level feature statistics:")
print(participant_features_df.describe())

# Correlation matrix
correlation_matrix = participant_features_df.corr()

plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Participant-Level Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# =============================================================================

# Scale features
scaler = StandardScaler()
participant_features_scaled = scaler.fit_transform(participant_features_df)

# Perform K-means clustering
nclusters = 3
kmeans = KMeans(n_clusters=nclusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(participant_features_scaled)

# Add cluster labels to participant features DataFrame
participant_features_df['cluster'] = cluster_labels

print("\nCluster distribution:")
print(participant_features_df['cluster'].value_counts().sort_index())

# =============================================================================
# Apply PCA
pca = PCA()
participant_features_pca = pca.fit_transform(participant_features_scaled)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("PCA Explained Variance Ratio (first 5 components):")
for i, (var, cum_var) in enumerate(zip(explained_variance[:5], cumulative_variance[:5])):
    print(f"PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")

# Plot explained variance
plt.figure(figsize=(12, 5))
plt.bar(range(1, len(explained_variance[:10]) + 1), explained_variance[:10], alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Explained Variance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize clusters using PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(participant_features_pca[:, 0], participant_features_pca[:, 1], 
                     c=cluster_labels, cmap='viridis', alpha=0.7, s=60)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Participant Clusters (PCA Visualization)')
plt.grid(True, alpha=0.3)

# Add cluster centers in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, 
           label='Cluster Centers', edgecolors='black')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
print("\nFEATURE - PCA COMPONENT RELATIONSHIPS")

# Get PCA component weights (loadings)
components = pca.components_

# Create loadings DataFrame
loadings_df = pd.DataFrame(components.T, 
                          columns=[f'PC{i+1}' for i in range(len(feature_columns))],
                          index=feature_columns)

print("Top features for PC1 and PC2:")

# PC1 top features
pc1_loadings = loadings_df['PC1'].abs().sort_values(ascending=False).head(10)
print("\nTop 10 features for PC1:")
for feature, loading in pc1_loadings.items():
    print(f"  {feature}: {loadings_df.loc[feature, 'PC1']:.3f}")

# PC2 top features
pc2_loadings = loadings_df['PC2'].abs().sort_values(ascending=False).head(10)
print("\nTop 10 features for PC2:")
for feature, loading in pc2_loadings.items():
    print(f"  {feature}: {loadings_df.loc[feature, 'PC2']:.3f}")

# Visualize feature contributions to first two components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# PC1 feature contributions
pc1_top_features = pc1_loadings.index[:8]
pc1_values = [loadings_df.loc[feature, 'PC1'] for feature in pc1_top_features]
ax1.barh(pc1_top_features, pc1_values)
ax1.set_xlabel('Loading Value')
ax1.set_title('Top Features Contributing to PC1')
ax1.grid(True, alpha=0.3)

# PC2 feature contributions
pc2_top_features = pc2_loadings.index[:8]
pc2_values = [loadings_df.loc[feature, 'PC2'] for feature in pc2_top_features]
ax2.barh(pc2_top_features, pc2_values)
ax2.set_xlabel('Loading Value')
ax2.set_title('Top Features Contributing to PC2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================

print("\nCLUSTER CHARACTERISTICS")

# Analyze cluster characteristics
cluster_characteristics = participant_features_df.groupby('cluster').mean()
print("\nCluster characteristics (mean values for top 10 most variable features):")

# Find features with highest variation across clusters
feature_variation = cluster_characteristics.std() / cluster_characteristics.mean()
top_variable_features = feature_variation.sort_values(ascending=False).head(10).index

print(cluster_characteristics[top_variable_features].round(3))

# Visualize cluster characteristics
plt.figure(figsize=(14, 8))
cluster_means_standardized = (cluster_characteristics - cluster_characteristics.mean()) / cluster_characteristics.std()
sns.heatmap(cluster_means_standardized[top_variable_features].T, annot=True, 
            cmap='coolwarm', center=0, fmt='.2f')
plt.title('Standardized Feature Means by Cluster\n(Top 10 Most Variable Features)')
plt.tight_layout()
plt.show()

# =============================================================================
print("\nPARTICIPANT PROFILES ANALYSIS")

# Analyze what distinguishes each cluster
print("\nKey differences between clusters:")

for cluster in range(nclusters):
    cluster_data = participant_features_df[participant_features_df['cluster'] == cluster]
    print(f"\nCluster {cluster} (n={len(cluster_data)}):")
    
    # Get top 5 features that characterize this cluster
    cluster_mean = cluster_data.mean()
    overall_mean = participant_features_df.mean()
    
    # Find features where this cluster differs most from overall mean
    differences = (cluster_mean - overall_mean).abs().sort_values(ascending=False)
    top_differences = differences.head(5)
    
    for feature, diff in top_differences.items():
        if feature != 'cluster':
            cluster_val = cluster_mean[feature]
            overall_val = overall_mean[feature]
            print(f"  {feature}: {cluster_val:.3f} (overall: {overall_val:.3f})")

# =============================================================================

print("\nCluster sizes:")

for cluster in range(nclusters):
    cluster_data = participant_features_df[participant_features_df['cluster'] == cluster]
    n_participants = len(cluster_data)
    percentage = (n_participants / len(participant_features_df)) * 100
    cluster_participants = cluster_data.index.tolist()
    
    print(f"\nCluster {cluster}: {n_participants} participants ({percentage:.1f}%)")
    print(f"Participants: {', '.join(cluster_participants)}")