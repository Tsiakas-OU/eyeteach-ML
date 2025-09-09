# generate explanations for models and predictions

## tasks
1. generate clustering summary - [code](../../clustering_trial.py):  
    - input: cluster summary (mean feature values per cluster)
    - output: summary report for clustering

2. generate local explanations 
    - input: 
        - SHAP values - - [code](../../explanations/shap_explanations.py)
        - test_sample: the feature vector used for the prediction
    - output: explanation-based report for the specific prediction

2. generate global explanations 
    - input: 
        - SHAP values - - [code](../../explanations/shap_explanations.py)
        - test_data: new data (not used in training)
    - output: explanation-based report for how the model makes decisions