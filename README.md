# [EYE-TEACH] Machine Learning models
![Static Badge](https://img.shields.io/badge/status-work_in_progress-orange) 

**🚧 Under Active Development | Draft Version | Pre-Release** 

Development of ML models using eye-tracking data for reading comprehension tasks 

## Attribution
For the purpose of the EYE-TEACH project, we sourced existing datasets and codes from:
- [MECO-L1](https://osf.io/3527a/), under the [CC-By Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license 
- [WebQAmGaze](https://github.com/tfnribeiro/WebQAmGaze), under the [CC-By Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license  

## Installation
- requirements - see `requirements.txt` or `environment.yml`
- if needed, create/activate virtual environment (e.g., conda or venv)  

## Folders and Files
- `datasets/`: training datasets for model training
- `descriptions/`: descriptions of datasets, models, and teacher insights
- `visualizations/`: codes for data visualizations
- `explanations/`: codes for shap values and LLM-based explanations
- `training/`: codes for model training
- `trained_models/`: trained models
- `classifier_trial_meco.py`: classification of comprehension accuracy - MECO passage level
- `clustering_trial_meco.py`: clustering of eye-tracking features and comprehension data - MECO passage level

## Training datasets
Based on the existing datasets mentioned above, for this project we use: 
- `MECO-en_uk-passage.csv`: MECO-L1 passage data for English (UK). It includes ET features and reading comprehension scores [0-4] for each trial [1-12] - [see more](descriptions/data/mecoL1.md)
- `MECO-en_uk-fix_uk01.csv`: MECO-L1 fixation report for a single participant
- `MECO-en_uk-comp.csv`: MECO-L1 reading comprehension scores [0-4] for each passage/trial [1-12]
- `webq-nr.csv`: Data from WebQAmGaze - used code from https://github.com/tfnribeiro/WebQAmGaze to get the csv data from the pre_processed_data folder

## Methods - Visualizations and Explanations
- plot saccades path for a single participant's trial - MECO L1 - (`visualizations\saccade_plot_meco.py`)
- plot fixation path for a single participant's trial - MECO L1 - (`visualizations\saccade_plot_meco.py`)
- (Draft) visualization of LLM-based cluster analysis - open `visualizations\cluster_analysis_example.html`
- SHAP explanations - global and local explanations based on SHAP values (`explanations\shap_explanations.py`)
- LLM feedback - communicate model parameters and explanations (`explanations\LLM_explanations.py`)
    - provide a descriptive summary of clustering (`explanations\generate_cluster_explanations`)
    - provide global/local explanations based on SHAP explanations 


## Usage and Tasks
### Tasks
- Predict reading comprehension - [description](descriptions/models/predict.md)
- Cluster eye tracking and reading comprehension data -[description](descriptions/models/cluster.md)

### Usage
Run `python` + 
- `training\classifier_trial_meco.py` - [MECO] train a classifier to predict the reading comprehension score for a single participant's trials (passage level)
- `training\clustering_trial_meco.py` - [MECO] identify clusters of trial data (passage level)
- `visualizations\saccade_plot_meco.py` - [MECO] plot saccades for a participant's trial
 
