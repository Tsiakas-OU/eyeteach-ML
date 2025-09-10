#from transformers import pipeline
#import json
#import torch

"""
pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=400  
)
"""

def generate_cluster_explanations(cluster_profiles):
    """Generate natural language explanations for all clusters with comparative analysis"""
    
    explanations = {}
    
    # Prepare cluster data for all clusters
    cluster_data = {}
    for cluster_id in cluster_profiles.index:
        cluster_data[cluster_id] = cluster_profiles.loc[cluster_id].to_dict()
    
    # Create a comprehensive prompt for analysis
    prompt = f"""
    Analyze these three eye-tracking data clusters from a reading comprehension study and provide a cluster analysis:
    
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

    Please provide a short analysis including one to two sentences for:
    1. Key characteristics that distinguish each group
    2. Comparative analysis of how the clusters differ
    3. How each pattern relates to reading comprehension outcomes
    4. Overall patterns and insights across all clusters

    Analyze these reader profiles from an eye-tracking study. 
    Describe each group's reading style in simple terms, compare their approaches, 
    and explain how each style related to their reading comprehension score. 
    Keep the analysis objective and teacher-friendly, avoiding technical jargon like 'fixations' or 'regressions'.
    """
    
    try:
        # Generate cluster analysis -NOW IT PROVIDES A STATIC RESPONSE - LLM generated
        print(prompt)
        """ response = pipe(
            prompt,
            max_new_tokens=500,
            temperature=0.2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )[0]['generated_text'] 
        
        # Extract only the new generated text
        #cluster_analysis = response.replace(prompt, "").strip()
        #explanations['clusters'] = cluster_analysis
        
        #print("\nCluster analysis:")
        #print(cluster_analysis)
        #print("-" * 80)
        """
        static_reponse = f"Analysis of Three Reader Groups\nWe can see three clear patterns in how students read.\n One group reads at a medium pace but goes back to reread sections quite often. Another group reads very quickly and smoothly, rarely going back over text they have already read. The final group reads very slowly and spends a significant amount of time rereading previous words and sentences. \n Despite these very different approaches to reading, all three groups showed very similar and high levels of understanding on the comprehension questions afterward.\n The main difference between the groups is the speed and effort spent reading, not the final level of understanding they achieved."
        print(static_reponse)

    except Exception as e:
        print(f"Error generating explanation for cluster {cluster_id}: {e}")
    
    return explanations
