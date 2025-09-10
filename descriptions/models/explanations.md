# generate explanations for models and predictions

## tasks
1. generate LLM-based clustering summary - [code](../../explanations/LLM_explanations.py):  
    - input: cluster summary (mean feature values per cluster)
    - output: summary report for clustering

**Example:** 
*We can see three clear patterns in how students read. One group reads at a medium pace but goes back to reread sections quite often. Another group reads very quickly and smoothly, rarely going back over text they have already read. The final group reads very slowly and spends a significant amount of time rereading previous words and sentences. Despite these very different approaches to reading, all three groups showed very similar and high levels of understanding on the comprehension questions afterward. The main difference between the groups is the speed and effort spent reading, not the final level of understanding they achieved.*

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