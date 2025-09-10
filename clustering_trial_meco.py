import pandas as pd
from explanations.LLM_explanations import generate_cluster_explanations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('datasets/mecoL1/MECO-en_uk-passage.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Define features
features = ['nblink', 'nrun', 'nfix', 'nout', 
           'sac', 'skip', 'refix', 'reg', 'mfix', 
           'firstpass', 'rereading', 'total', 'rate', 'ACCURACY']

print(f"\nFeatures for clustering ({len(features)}): {features}")

# Check for missing values and clean data
print("\nMissing values in features:")
print(df[features].isnull().sum())

# Remove rows with missing values
df_clean = df.dropna(subset=features)
print(f"Clean data shape: {df_clean.shape}")

# Scale the features (including accuracy)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = clusters

print(f"\nCluster distribution:")
cluster_counts = df_clean['cluster'].value_counts().sort_index()
print(cluster_counts)

# Calculate correlation between features and cluster assignments
feature_correlations = {}
for feature in features:
    correlation = np.corrcoef(df_clean['cluster'], df_clean[feature])[0, 1]
    feature_correlations[feature] = abs(correlation)  # Use absolute value for importance

# Sort features by correlation strength
sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)

print("Features most correlated with cluster assignments:")
for feature, corr in sorted_features[:5]:  # Top 5 features
    print(f"{feature}: {corr:.3f}")

# Detailed cluster profiles
print("\n=== CLUSTER PROFILES ===")
cluster_profiles = df_clean.groupby('cluster')[features].mean().round(2)
print("Mean feature values by cluster:")
print(cluster_profiles)

## get LLM-based explanations
explanations =  generate_cluster_explanations(cluster_profiles)


