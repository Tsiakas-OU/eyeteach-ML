from transformers import pipeline
import json
import torch


pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=400  
)


def generate_cluster_explanations(cluster_profiles):
    """Generate natural language explanations for all clusters with comparative analysis"""
    
    explanations = {}
    
    # Prepare cluster data for all clusters
    cluster_data = {}
    for cluster_id in cluster_profiles.index:
        cluster_data[cluster_id] = cluster_profiles.loc[cluster_id].to_dict()
    
    # Create a comprehensive prompt for comparative analysis
    prompt = f"""
    Analyze these three eye-tracking data clusters from a reading comprehension study and provide a comprehensive comparative analysis:
    
    CLUSTER 0:
    - Number of blinks: {cluster_data[0]['nblink']}
    - Number of fixations: {cluster_data[0]['nfix']}
    - Reading rate: {cluster_data[0]['rate']} words per minute
    - Rereading time: {cluster_data[0]['rereading']} milliseconds
    - Comprehension accuracy: {cluster_data[0]['ACCURACY']}/4.0
    - Regression frequency: {cluster_data[0]['reg']}
    - Total reading time: {cluster_data[0]['total']} milliseconds
    
    CLUSTER 1:
    - Number of blinks: {cluster_data[1]['nblink']}
    - Number of fixations: {cluster_data[1]['nfix']}
    - Reading rate: {cluster_data[1]['rate']} words per minute
    - Rereading time: {cluster_data[1]['rereading']} milliseconds
    - Comprehension accuracy: {cluster_data[1]['ACCURACY']}/4.0
    - Regression frequency: {cluster_data[1]['reg']}
    - Total reading time: {cluster_data[1]['total']} milliseconds
    
    CLUSTER 2:
    - Number of blinks: {cluster_data[2]['nblink']}
    - Number of fixations: {cluster_data[2]['nfix']}
    - Reading rate: {cluster_data[2]['rate']} words per minute
    - Rereading time: {cluster_data[2]['rereading']} milliseconds
    - Comprehension accuracy: {cluster_data[2]['ACCURACY']}/4.0
    - Regression frequency: {cluster_data[2]['reg']}
    - Total reading time: {cluster_data[2]['total']} milliseconds

    Please provide a comprehensive analysis including:
    1. Descriptive labels for each reading pattern cluster
    2. Key characteristics that distinguish each group
    3. Comparative analysis of how the clusters differ
    4. How each pattern relates to reading comprehension outcomes
    5. Educational implications and teaching recommendations for each cluster
    6. Overall patterns and insights across all clusters

    Structure your response with clear sections and focus on comparative insights.
    Provide the response in a structured format without markdown.
    """
    
    try:
        # Generate comprehensive comparative analysis
        print(prompt)
        response = pipe(
            prompt,
            max_new_tokens=500,
            temperature=0.4,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )[0]['generated_text']
        
        # Extract only the new generated text
        comprehensive_analysis = response.replace(prompt, "").strip()
        explanations['comprehensive'] = comprehensive_analysis
        
        print("\nCOMPREHENSIVE COMPARATIVE ANALYSIS:")
        print(comprehensive_analysis)
        print("-" * 80)
    
            
    except Exception as e:
        print(f"Error generating explanation for cluster {cluster_id}: {e}")
    
    return explanations
