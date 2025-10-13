import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
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

# Initialize lists to store results
all_predictions = []
all_true_labels = []
trial_results = []
feature_importances = []
selected_features_per_fold = []  # Track which features were selected in each fold

print("\nStarting Leave-One-Trial-Out Cross-Validation...")
print("=" * 60)

# Perform Leave-One-Trial-Out Cross-Validation
# train the model on N-1 passages and test in on a new passage
for i, test_trial in enumerate(trials):
    print(f"Fold {i+1}/{len(trials)}: Testing trial {test_trial}")
    
    # Split data: all trials except the current one for training
    train_mask = df['trialid'] != test_trial
    test_mask = df['trialid'] == test_trial
    
    X_train = df.loc[train_mask, all_features]
    y_train = df.loc[train_mask, 'ACCURACY_CLASS']
    X_test = df.loc[test_mask, all_features]
    y_test = df.loc[test_mask, 'ACCURACY_CLASS']

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42 + i)
    try:
        X_train_b, y_train_b = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE - Train samples: {len(X_train)} → {len(X_train_b)}")
    except ValueError as e:
        print(f"  SMOTE failed: {e}, using original data")
        X_train_b, y_train_b = X_train, y_train
    
    # Check if test trial has data
    if len(X_test) == 0:
        print(f"  No data for trial {test_trial}, skipping...")
        continue
    
    # Check if enough samples for training
    if len(X_train) <= 40:
        print(f"  No training data available for trial {test_trial}, skipping...")
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
    
    # Select features above importance threshold
    #importance_threshold = 0.07  # You can adjust this threshold
    #selected_features_mask = selector_rf.feature_importances_ >= importance_threshold
    #selected_features = [all_features[j] for j in range(len(all_features)) if selected_features_mask[j]]
    
    # Alternative: Select top k features
    k_features = 5
    selected_features = feature_importance_selector.head(k_features)['feature'].tolist()
    selected_features_mask = [feature in selected_features for feature in all_features]
    
    print(f"  Selected {len(selected_features)} features: {selected_features}")
    print(f"  Feature importances: {dict(zip(feature_importance_selector['feature'], feature_importance_selector['importance'].round(3)))}")
    
    # Apply feature selection
    X_train_selected = X_train_scaled[:, selected_features_mask]
    X_test_selected = X_test_scaled[:, selected_features_mask]
    
    # Store which features were selected in this fold
    selected_features_per_fold.append({
        'trial': test_trial,
        'selected_features': selected_features,
        'feature_importances': dict(zip(all_features, selector_rf.feature_importances_.round(4)))
    })
    
    ## If not SMOTE: Handle class imbalance by computing class weights
    #classes = np.unique(y_train)
    #class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    #class_weight_dict = dict(zip(classes, class_weights))
    
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
    
    # Calculate accuracy for this trial
    trial_accuracy = accuracy_score(y_test, y_pred)
    
    # Store feature importances (for selected features only)
    feature_imp = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_,
        'trial': test_trial
    })
    feature_importances.append(feature_imp)
    
    # Store trial-level results
    trial_results.append({
        'trial_id': test_trial,
        'test_samples': len(X_test),
        'train_samples': len(X_train),
        'selected_features': selected_features,
        'n_features_selected': len(selected_features),
        'accuracy': trial_accuracy,
        'true_labels': y_test.values.tolist(),
        'predictions': y_pred.tolist(),
        'participants_in_test': df.loc[test_mask, 'uniform_id'].unique().tolist()
    })
    
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {len(selected_features)}, Accuracy: {trial_accuracy:.3f}")

# Calculate overall performance
print("\n" + "=" * 60)
print("OVERALL RESULTS - Leave-One-Trial-Out with RF Feature Selection")
print("=" * 60)

overall_accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Overall Accuracy: {overall_accuracy:.3f}")
print(f"Total predictions: {len(all_predictions)}")
print(f"Number of trials tested: {len(trials)}")

# classification report
print("\nClassification Report:")
print(classification_report(all_true_labels, all_predictions))

# Create confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions)

print("Confusion Matrix - 3 Labels")
print("=" * 50)
cm_df = pd.DataFrame(cm)
print(cm_df)

# Trial-level performance summary
trial_summary = pd.DataFrame(trial_results)
print(f"\nTrial-level Performance Summary:")
print(f"Average accuracy across trials: {trial_summary['accuracy'].mean():.3f}")
print(f"Standard deviation: {trial_summary['accuracy'].std():.3f}")
print(f"Best trial accuracy: {trial_summary['accuracy'].max():.3f}")
print(f"Worst trial accuracy: {trial_summary['accuracy'].min():.3f}")
print(f"Average features selected: {trial_summary['n_features_selected'].mean():.1f}")

balanced_acc = balanced_accuracy_score(all_true_labels, all_predictions)
f1 = f1_score(all_true_labels, all_predictions, average='weighted')

print(f"Balanced Accuracy: {balanced_acc:.3f}")
print(f"F1 Score: {f1:.3f}")

# Create accuracy plot for each trial/fold
plt.figure(figsize=(14, 8))
trials_plot = [f"Trial {result['trial_id']}" for result in trial_results]
accuracies = [result['accuracy'] for result in trial_results]

# Color points based on performance - compare to baseline
#colors = ['green' if acc >= trial_summary['accuracy'].mean() else 'red' for acc in accuracies]
colors = ['green' if acc > 0.33 else 'red' for acc in accuracies]

bars = plt.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
plt.axhline(y=trial_summary['accuracy'].mean(), color='blue', linestyle='--', 
            label=f'Mean Accuracy: {trial_summary["accuracy"].mean():.3f}')
plt.axhline(y=trial_summary['accuracy'].mean() + trial_summary['accuracy'].std(), 
            color='orange', linestyle=':', alpha=0.7, label='±1 Std Dev')
plt.axhline(y=trial_summary['accuracy'].mean() - trial_summary['accuracy'].std(), 
            color='orange', linestyle=':', alpha=0.7)

plt.xlabel('Trial Fold')
plt.ylabel('Accuracy')
plt.title('Leave-One-Trial-Out Cross-Validation: Accuracy per Fold')
plt.xticks(range(len(accuracies)), [f'{i+1}' for i in range(len(accuracies))], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

if feature_importances:
    feature_importance_df = pd.concat(feature_importances, ignore_index=True)
    avg_feature_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print(f"\nAverage Feature Importance:")
    for feature, importance in avg_feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    plt.figure(figsize=(10, 6))
    avg_feature_importance.plot(kind='bar')
    plt.title('Average Feature Importance Across All Trials')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()