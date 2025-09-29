import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv('../datasets/mecoL1/MECO-en_uk-passage.csv')
print(df.head())

participants = df['uniform_id'].unique()
print(f"Total participants: {len(participants)}")

trials = df['trialid'].unique()
print(f"Total trials: {len(trials)}")

# Define features (X) and target (y)
features = ['trial.nwords', 'nblink', 'nrun', 'nfix', 'nout', 
           'sac', 'skip', 'refix', 'reg', 'mfix', 
           'firstpass', 'rereading', 'total', 'rate']

# Define new classes
def recode_accuracy(x):
    if x in [0, 1, 2]:
        return 'Low_Medium'  # Class 0-2 merged
    elif x == 3:
        return 'High'        # Class 3 kept
    elif x == 4:
        return 'Perfect'   # Class 4 kept

df['ACCURACY_CLASS'] = df['ACCURACY'].apply(recode_accuracy)
print(df['ACCURACY_CLASS'].value_counts())

# Initialize lists to store results
all_predictions = []
all_true_labels = []
participant_results = []
feature_importances = []

print("\nStarting Leave-One-Subject-Out Cross-Validation...")
print("=" * 60)

# Perform Leave-One-Subject-Out Cross-Validation
for i, test_participant in enumerate(participants):
    print(f"Fold {i+1}/{len(participants)}: Testing participant {test_participant}")
    
    # Split data: all participants except the current one for training
    train_mask = df['uniform_id'] != test_participant
    test_mask = df['uniform_id'] == test_participant
    
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, 'ACCURACY_CLASS']
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, 'ACCURACY_CLASS']
    
    # Check if test participant has data
    if len(X_test) == 0:
        print(f"  No data for participant {test_participant}, skipping...")
        continue
    
    # Check if we have enough samples for training
    if len(X_train) == 0:
        print(f"  No training data available for participant {test_participant}, skipping...")
        continue
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance by computing class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Train Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Store results
    all_predictions.extend(y_pred)
    all_true_labels.extend(y_test.values)
    
    # Calculate accuracy for this participant
    participant_accuracy = accuracy_score(y_test, y_pred)
    
    # Store feature importances
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_,
        'participant': test_participant
    })
    feature_importances.append(feature_imp)
    
    # Store participant-level results
    participant_results.append({
        'participant_id': test_participant,
        'test_samples': len(X_test),
        'train_samples': len(X_train),
        'accuracy': participant_accuracy,
        'true_labels': y_test.values.tolist(),
        'predictions': y_pred.tolist(),
        'trials_in_test': df.loc[test_mask, 'trialid'].unique().tolist()
    })
    
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}, Accuracy: {participant_accuracy:.3f}")

# Calculate overall performance
print("\n" + "=" * 60)
print("OVERALL RESULTS - Leave-One-Subject-Out")
print("=" * 60)

overall_accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Overall Accuracy: {overall_accuracy:.3f}")
print(f"Total predictions: {len(all_predictions)}")
print(f"Number of participants tested: {len(participants)}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(all_true_labels, all_predictions))

# Participant-level performance summary
participant_summary = pd.DataFrame(participant_results)
print(f"\nParticipant-level Performance Summary:")
print(f"Average accuracy across participants: {participant_summary['accuracy'].mean():.3f}")
print(f"Standard deviation: {participant_summary['accuracy'].std():.3f}")
print(f"Best participant accuracy: {participant_summary['accuracy'].max():.3f}")
print(f"Worst participant accuracy: {participant_summary['accuracy'].min():.3f}")

# Feature importance analysis
if feature_importances:
    feature_importance_df = pd.concat(feature_importances, ignore_index=True)
    avg_feature_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print(f"\nAverage Feature Importance:")
    for feature, importance in avg_feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    avg_feature_importance.plot(kind='bar')
    plt.title('Average Feature Importance Across All Participants')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()