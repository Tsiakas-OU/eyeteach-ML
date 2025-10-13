import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('../datasets/mecoL1/MECO-en_uk-passage.csv')
print(df.head())

# Load Flesch scores
flesch_df = pd.read_csv('../datasets/mecoL1/flesch_scores_eng_MECO_L1.csv')
print(flesch_df.head())
flesch_dict = flesch_df.set_index('trialid')['flesch_readability'].to_dict()

# Map flesch_readability to main dataset
df['flesch_readability'] = df['trialid'].map(flesch_dict)
print(df.head())

participants = df['uniform_id'].unique()
print(f"Total participants: {len(participants)}")

trials = df['trialid'].unique()
print(f"Total trials: {len(trials)}")
print(df['ACCURACY'].value_counts())

# All available features
all_features = ['nblink', 'nrun', 'nfix', 'nout', 'trial.nwords',
                'sac', 'skip', 'refix', 'reg', 'mfix', 
                'firstpass', 'rereading', 'total', 'rate', 'flesch_readability']

# Define new classes
def recode_accuracy(x):
    if x in [0, 1, 2]:
        return 0  # Class 0-2 merged
    elif x == 3:
        return 1  # Class 3 kept
    elif x == 4:
        return 2  # Class 4 kept

df['ACCURACY_CLASS'] = df['ACCURACY'].apply(recode_accuracy).astype(int)
print(df['ACCURACY_CLASS'].value_counts())

# =============================================================================
# STEP 1: Cluster participants based on their eye-tracking patterns (MEANS ONLY)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: Clustering Participants (Feature Means Only)")
print("=" * 60)

# Create participant-level features using ONLY MEANS
participant_features = []
for participant in participants:
    participant_data = df[df['uniform_id'] == participant]
    
    # Calculate mean for each feature across all trials
    feature_vector = []
    for feature in all_features:
        feature_vector.append(participant_data[feature].mean())  # MEAN ONLY
    
    participant_features.append(feature_vector)

# Convert to DataFrame
participant_features_df = pd.DataFrame(participant_features, index=participants)
feature_columns = [f'{feature}_mean' for feature in all_features]
participant_features_df.columns = feature_columns

print(f"Created participant feature matrix: {participant_features_df.shape}")

# Scale features for clustering
cluster_scaler = StandardScaler()
participant_features_scaled = cluster_scaler.fit_transform(participant_features_df)

# Perform K-means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(participant_features_scaled)

# Add cluster labels to participant features DataFrame
participant_features_df['cluster'] = cluster_labels

# Add cluster labels to the main DataFrame
cluster_mapping = dict(zip(participants, cluster_labels))
df['cluster'] = df['uniform_id'].map(cluster_mapping)

print("\nCluster distribution:")
cluster_counts = participant_features_df['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    percentage = (count / len(participant_features_df)) * 100
    print(f"  Cluster {cluster}: {count} participants ({percentage:.1f}%)")

# =============================================================================
# STEP 2: Cluster-Specific Classification with Leave-One-Subject-Out
# =============================================================================

# Initialize lists to store results
all_predictions = []
all_true_labels = []
participant_results = []
feature_importances = []
selected_features_per_fold = []  # Track which features were selected in each fold

print("\nStarting Leave-One-Subject-Out Cross-Validation with Cluster-Specific Models...")
print("=" * 60)

for i, test_participant in enumerate(participants):
    print(f"Fold {i+1}/{len(participants)}: Testing participant {test_participant}")
    
    # Get cluster of test participant
    test_cluster = cluster_mapping[test_participant]
    
    # Split data: all participants except the current one for training
    train_mask = df['uniform_id'] != test_participant
    test_mask = df['uniform_id'] == test_participant
    
    # Use only training data from the same cluster as test participant
    cluster_train_mask = train_mask & (df['cluster'] == test_cluster)
    
    X_train = df.loc[cluster_train_mask, all_features]
    y_train = df.loc[cluster_train_mask, 'ACCURACY_CLASS']
    X_test = df.loc[test_mask, all_features]
    y_test = df.loc[test_mask, 'ACCURACY_CLASS']
    
    # Check if we have enough samples for training
    if len(X_train) == 0:
        print(f"  No training data in cluster {test_cluster}, using all clusters...")
        # Fallback: use all training data if cluster-specific data is insufficient
        X_train = df.loc[train_mask, all_features]
        y_train = df.loc[train_mask, 'ACCURACY_CLASS']
        used_fallback = True
    else:
        used_fallback = False
    
    if len(X_test) == 0:
        print(f"  No test data for participant {test_participant}, skipping...")
        continue
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42 + i)
    try:
        X_train_b, y_train_b = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE - Train samples: {len(X_train)} → {len(X_train_b)}")
    except ValueError as e:
        print(f"  SMOTE failed: {e}, using original data")
        X_train_b, y_train_b = X_train, y_train
    
    # Check if enough samples for training
    if len(X_train) <= 10:
        print(f"  Insufficient training data for participant {test_participant}, skipping...")
        continue
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_b)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest to get feature importance for selection
    selector_rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42 + i,
        n_jobs=-1
    )
    
    selector_rf.fit(X_train_scaled, y_train_b)
    
    # Get feature importance and select top features
    feature_importance_selector = pd.DataFrame({
        'feature': all_features,
        'importance': selector_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top k features
    k_features = 5
    selected_features = feature_importance_selector.head(k_features)['feature'].tolist()
    selected_features_mask = [feature in selected_features for feature in all_features]
    
    print(f"  Selected {len(selected_features)} features: {selected_features}")
    
    # Apply feature selection
    X_train_selected = X_train_scaled[:, selected_features_mask]
    X_test_selected = X_test_scaled[:, selected_features_mask]
    
    # Store which features were selected in this fold
    selected_features_per_fold.append({
        'participant': test_participant,
        'cluster': test_cluster,
        'selected_features': selected_features,
        'feature_importances': dict(zip(all_features, selector_rf.feature_importances_.round(4))),
        'used_fallback': used_fallback
    })
    
    # Train Final Random Forest classifier (on selected features)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_selected, y_train_b)
    
    # Make predictions
    y_pred = model.predict(X_test_selected)
    
    # Store results
    all_predictions.extend(y_pred)
    all_true_labels.extend(y_test.values)
    
    # Calculate accuracy for this participant
    participant_accuracy = accuracy_score(y_test, y_pred)
    
    # Store feature importances (for selected features only)
    feature_imp = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_,
        'participant': test_participant,
        'cluster': test_cluster
    })
    feature_importances.append(feature_imp)
    
    # Store participant-level results
    participant_results.append({
        'participant_id': test_participant,
        'cluster': test_cluster,
        'test_samples': len(X_test),
        'train_samples': len(X_train),
        'train_samples_same_cluster': len(df.loc[cluster_train_mask]),
        'selected_features': selected_features,
        'n_features_selected': len(selected_features),
        'used_fallback': used_fallback,
        'accuracy': participant_accuracy,
        'true_labels': y_test.values.tolist(),
        'predictions': y_pred.tolist(),
        'trials_in_test': df.loc[test_mask, 'trialid'].unique().tolist()
    })
    
    print(f"  Cluster: {test_cluster}, Train samples: {len(X_train)} (same cluster: {len(df.loc[cluster_train_mask])}), Features: {len(selected_features)}, Accuracy: {participant_accuracy:.3f}")

# =============================================================================
# STEP 3: Comprehensive Results Analysis
# =============================================================================

print("\n" + "=" * 60)
print("OVERALL RESULTS - Cluster-Specific Classification with Feature Selection")
print("=" * 60)

overall_accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Overall Accuracy: {overall_accuracy:.3f}")
print(f"Total predictions: {len(all_predictions)}")
print(f"Number of participants tested: {len(participant_results)}")

# Classification report
print("\nClassification Report:")
print(classification_report(all_true_labels, all_predictions))

# Create confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions)
print("\nConfusion Matrix - 3 Labels")
print("=" * 50)
cm_df = pd.DataFrame(cm)
print(cm_df)

# Participant-level performance summary
participant_summary = pd.DataFrame(participant_results)
print(f"\nParticipant-level Performance Summary:")
print(f"Average accuracy across participants: {participant_summary['accuracy'].mean():.3f}")
print(f"Standard deviation: {participant_summary['accuracy'].std():.3f}")
print(f"Best participant accuracy: {participant_summary['accuracy'].max():.3f}")
print(f"Worst participant accuracy: {participant_summary['accuracy'].min():.3f}")
print(f"Average features selected: {participant_summary['n_features_selected'].mean():.1f}")

# Additional metrics
balanced_acc = balanced_accuracy_score(all_true_labels, all_predictions)
f1 = f1_score(all_true_labels, all_predictions, average='weighted')

print(f"Balanced Accuracy: {balanced_acc:.3f}")
print(f"F1 Score (weighted): {f1:.3f}")

# Performance by cluster
print(f"\nPerformance by Cluster:")
cluster_performance = participant_summary.groupby('cluster').agg({
    'accuracy': ['mean', 'std', 'count'],
    'n_features_selected': 'mean',
    'used_fallback': 'sum'
}).round(3)
print(cluster_performance)

# =============================================================================
# VISUALIZATIONS
# =============================================================================

# Create accuracy plot for each participant/fold
plt.figure(figsize=(14, 8))
participants_plot = [f"Part {result['participant_id']}" for result in participant_results]
accuracies = [result['accuracy'] for result in participant_results]
clusters = [result['cluster'] for result in participant_results]

# Color points based on cluster
colors = ['green' if cluster == 0 else 'blue' for cluster in clusters]

bars = plt.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
plt.axhline(y=participant_summary['accuracy'].mean(), color='red', linestyle='--', 
            label=f'Mean Accuracy: {participant_summary["accuracy"].mean():.3f}')
plt.axhline(y=participant_summary['accuracy'].mean() + participant_summary['accuracy'].std(), 
            color='orange', linestyle=':', alpha=0.7, label='±1 Std Dev')
plt.axhline(y=participant_summary['accuracy'].mean() - participant_summary['accuracy'].std(), 
            color='orange', linestyle=':', alpha=0.7)

plt.xlabel('Participant Fold')
plt.ylabel('Accuracy')
plt.title('Leave-One-Participant-Out Cross-Validation: Accuracy per Fold (Colored by Cluster)')
plt.xticks(range(len(accuracies)), [f'{i+1}' for i in range(len(accuracies))], rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot:cite[1]
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.show()

# Feature importance analysis
print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

if feature_importances:
    feature_importance_df = pd.concat(feature_importances, ignore_index=True)
    avg_feature_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print(f"\nAverage Feature Importance in Final Models:")
    for feature, importance in avg_feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    avg_feature_importance.head(10).plot(kind='bar')
    plt.title('Average Feature Importance Across All Participants')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Feature importance by cluster
if feature_importances:
    print(f"\nAverage Feature Importance by Cluster:")
    feature_importance_df = pd.concat(feature_importances, ignore_index=True)
    cluster_feature_importance = feature_importance_df.groupby(['cluster', 'feature'])['importance'].mean().unstack('cluster')
    
    # Plot feature importance by cluster
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for cluster_num in range(2):
        cluster_imp = feature_importance_df[feature_importance_df['cluster'] == cluster_num]
        avg_imp = cluster_imp.groupby('feature')['importance'].mean().sort_values(ascending=False).head(8)
        
        axes[cluster_num].bar(range(len(avg_imp)), avg_imp.values, color='skyblue', alpha=0.7)
        axes[cluster_num].set_title(f'Cluster {cluster_num} - Top Feature Importance')
        axes[cluster_num].set_xticks(range(len(avg_imp)))
        axes[cluster_num].set_xticklabels(avg_imp.index, rotation=45)
        axes[cluster_num].set_ylabel('Importance')
        axes[cluster_num].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Cluster characteristics analysis
print("\n" + "=" * 60)
print("CLUSTER CHARACTERISTICS ANALYSIS")
print("=" * 60)

# Analyze what distinguishes the participant clusters
print("\nParticipant Cluster Characteristics (Mean Values):")
for cluster in range(2):
    cluster_participants = participant_features_df[participant_features_df['cluster'] == cluster]
    other_cluster = 1 if cluster == 0 else 0
    other_participants = participant_features_df[participant_features_df['cluster'] == other_cluster]
    
    print(f"\nCluster {cluster} participants (n={len(cluster_participants)}):")
    
    # Calculate standardized differences for top features
    differences = {}
    for feature in feature_columns[:8]:  # Show first 8 features for brevity
        effect_size = abs(cluster_participants[feature].mean() - other_participants[feature].mean()) / participant_features_df[feature].std()
        differences[feature] = effect_size
    
    top_differences = sorted(differences.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("  Most distinctive features (effect size):")
    for feature, effect in top_differences:
        direction = "higher" if cluster_participants[feature].mean() > other_participants[feature].mean() else "lower"
        print(f"    {feature}: {effect:.2f} ({direction})")