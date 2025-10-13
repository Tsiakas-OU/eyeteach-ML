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

features = ['trial.nwords', 'nblink', 'nrun', 'nfix', 'nout', 
           'sac', 'skip', 'refix', 'reg', 'mfix', 
           'firstpass', 'rereading', 'total', 'rate', 'flesch_readability', 'ACCURACY']

# PARTICIPANT-LEVEL AGGREGATION (MEAN VALUES)
participants = df['uniform_id'].unique()
print(f"Total participants: {len(participants)}")

# Create participant-level features - MEAN for clustering, STD for analysis
participant_means = []
participant_stds = []
participant_ids = []

for participant in participants:
    participant_data = df[df['uniform_id'] == participant]
    mean_vector = []
    std_vector = []
    for feature in features:
        mean_vector.append(participant_data[feature].mean())
        std_vector.append(participant_data[feature].std())
    
    participant_means.append(mean_vector)
    participant_stds.append(std_vector)
    participant_ids.append(participant)

participant_means_df = pd.DataFrame(participant_means, index=participant_ids)
mean_columns = [f'{feature}_mean' for feature in features]
participant_means_df.columns = mean_columns

participant_stds_df = pd.DataFrame(participant_stds, index=participant_ids)
std_columns = [f'{feature}_std' for feature in features]
participant_stds_df.columns = std_columns

print(f"Created participant MEAN features: {participant_means_df.shape}")
print(f"Created participant STD features: {participant_stds_df.shape}")
participant_means_clean = participant_means_df.dropna()
print(f"Participants after removing missing values: {len(participant_means_clean)}")

# =============================================================================
# Create distribution plots for all features
# Split features into two groups of 8
features_group1 = mean_columns[:8]   # First 8 features
features_group2 = mean_columns[8:]   # Last 8 features

# Figure 1: First 8 features
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
axes1 = axes1.ravel()

for i, feature in enumerate(features_group1):
    axes1[i].hist(participant_means_clean[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
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
    axes2[i].hist(participant_means_clean[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes2[i].set_title(f'Distribution of {feature}', fontsize=12)
    axes2[i].set_xlabel(feature)
    axes2[i].set_ylabel('Frequency')
    axes2[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# PARTICIPANT-LEVEL STATISTICS
# =============================================================================
print("\n" + "="*60)
print("PARTICIPANT-LEVEL STATISTICS")
print("="*60)

# Statistics for MEAN features
print("Participant-level MEAN feature statistics:")
print(participant_means_clean.describe())

# Analysis with STD features
print("\nParticipant-level STD feature statistics (variability analysis):")
print(participant_stds_df.describe())

# Check participant consistency using STD features
consistency_scores = participant_stds_df.mean(axis=1)
print(f"\nParticipant consistency (average std across features):")
print(f"Most consistent participant: {consistency_scores.idxmin()} ({consistency_scores.min():.3f})")
print(f"Least consistent participant: {consistency_scores.idxmax()} ({consistency_scores.max():.3f})")

# Visualize consistency distribution
plt.figure(figsize=(10, 6))
plt.hist(consistency_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(consistency_scores.mean(), color='red', linestyle='--', 
            label=f'Mean: {consistency_scores.mean():.3f}')
plt.xlabel('Average Standard Deviation (Consistency Score)')
plt.ylabel('Number of Participants')
plt.title('Distribution of Participant Consistency Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Correlation matrix for participant-level MEAN features
correlation_matrix_means = participant_means_clean.corr()

plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix_means, dtype=bool))
sns.heatmap(correlation_matrix_means, mask=mask, annot=True, cmap='coolwarm', 
            center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8},
            annot_kws={'size': 8})
plt.title('Participant-Level Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


# PARTICIPANT-LEVEL PCA AND CLUSTERING
# Scale MEAN features for clustering
scaler = StandardScaler()
participant_means_scaled = scaler.fit_transform(participant_means_clean)

# Perform PCA on participant MEAN data
pca_participant = PCA()
participant_pca = pca_participant.fit_transform(participant_means_scaled)

# Explained variance
explained_variance_participant = pca_participant.explained_variance_ratio_
print(f"Explained variance by first 5 components:")
for i in range(5):
    print(f"  PC{i+1}: {explained_variance_participant[i]:.3f} ({explained_variance_participant[i]*100:.1f}%)")

# Show feature contributions to principal components
components = pca_participant.components_
print("\nTop MEAN features contributing to PC1:")
pc1_contributions = pd.Series(components[0], index=mean_columns).abs().sort_values(ascending=False)
for feature, loading in pc1_contributions.head(5).items():
    print(f"  {feature}: {loading:.3f}")

print("\nTop MEAN features contributing to PC2:")
pc2_contributions = pd.Series(components[1], index=mean_columns).abs().sort_values(ascending=False)
for feature, loading in pc2_contributions.head(5).items():
    print(f"  {feature}: {loading:.3f}")

# Apply K-means clustering on participant PCA data
n_clusters = 2
kmeans_participant = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
participant_cluster_labels = kmeans_participant.fit_predict(participant_means_scaled)  # Use scaled data!

print(f"\nParticipant cluster distribution:")
unique, counts = np.unique(participant_cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    percentage = (count / len(participant_cluster_labels)) * 100
    print(f"  Cluster {cluster}: {count} participants ({percentage:.1f}%)")

# Add cluster labels to the dataframe
participant_means_clean = participant_means_clean.copy()
participant_means_clean['cluster'] = participant_cluster_labels

plt.figure(figsize=(10, 8))
scatter = plt.scatter(participant_pca[:, 0], participant_pca[:, 1], 
                     c=participant_cluster_labels, cmap='viridis', alpha=0.7, s=60)
plt.xlabel(f'PC1 ({explained_variance_participant[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance_participant[1]:.2%} variance)')
plt.title('Participant Clustering (Mean Features)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Add cluster centers
centers_pca = pca_participant.transform(kmeans_participant.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, 
           label='Cluster Centers', edgecolors='black')
plt.legend()
plt.tight_layout()
plt.show()

# Analyze mean values for each cluster
cluster_means = participant_means_clean.groupby('cluster').mean()

print("\nCluster characteristics (mean values for features):")
print(cluster_means[mean_columns].round(3))

# Visualize cluster differences for top features
# Get top 6 most differentiating features using ANOVA
from sklearn.feature_selection import f_classif

f_values, p_values = f_classif(participant_means_scaled, participant_cluster_labels)
feature_anova = pd.DataFrame({'feature': mean_columns, 'f_value': f_values, 'p_value': p_values})
top_features = feature_anova.nlargest(6, 'f_value')['feature'].tolist()

# Create subplots for top features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    # Create boxplot for each feature by cluster
    cluster_data = []
    for cluster in range(n_clusters):
        cluster_data.append(participant_means_clean[participant_means_clean['cluster'] == cluster][feature])
    
    boxplot = axes[i].boxplot(cluster_data, labels=[f'Cluster {c}' for c in range(n_clusters)], 
                             patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[i].set_title(f'{feature}\n(F-value: {f_values[mean_columns.index(feature)]:.1f})')
    axes[i].set_ylabel('Value')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Top Differentiating Features Between Clusters', fontsize=16)
plt.tight_layout()
plt.show()

# Add cluster labels to STD dataframe for consistency analysis
participant_stds_clean = participant_stds_df.loc[participant_means_clean.index]
participant_stds_clean['cluster'] = participant_cluster_labels

# Analyze consistency by cluster
cluster_consistency = participant_stds_clean.groupby('cluster').mean().mean(axis=1)
print("Average consistency (std) by cluster:")
for cluster, consistency in cluster_consistency.items():
    print(f"  Cluster {cluster}: {consistency:.3f}")

# Calculate correlation between features and cluster assignments
feature_correlations = {}
for feature in mean_columns:
    correlation = np.corrcoef(participant_means_clean['cluster'], participant_means_clean[feature])[0, 1]
    feature_correlations[feature] = abs(correlation)
# Sort features by correlation strength
sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)

print("Features most correlated with cluster assignments:")
for feature, corr in sorted_features[:5]:
    print(f"  {feature}: {corr:.3f}")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS - IMPROVED
# =============================================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS (ANOVA F-values)")
print("="*60)

# Calculate F-values (variance between clusters / variance within clusters)
f_values, p_values = f_classif(participant_means_scaled, participant_cluster_labels)

feature_importance = pd.DataFrame({
    'feature': mean_columns,
    'f_value': f_values,
    'p_value': p_values,
    'significant': p_values < 0.05
}).sort_values('f_value', ascending=False)

print("Feature importance for cluster differentiation:")
print(feature_importance.head(10).round(4))