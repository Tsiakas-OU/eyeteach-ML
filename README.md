# [EYE-TEACH] Machine Learning models
development of ML models using eye-tracking data for reading comprehension tasks

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
- `classifier_trial.py`: classification of comprehension accuracy (passage level)
- `clustering_trial.py`: clustering of eye-tracking features and comprehension data (passage level)

## Data Used
- `MECO-en_uk-passage.csv`: MECO-L1 passage data for English (UK). It includes ET features and reading comprehension scores [0-4] for each passage/trial [1-12] - [see more](descriptions/data/mecoL1.md)
- 

## Methods ()
- plot saccades for a single participant's trial - (`visualizations\saccade_plot.py`)
- SHAP - global and local explanations (`explanations\shap_explanations.py`)
- LLM - communicate model parameters and explanations (`explanations\LLM_explanations.py`)
    - provide a descriptive summary of clustering (`explanations\generate_cluster_explanations`)
    - provide global/local explanations based on SHAP

## Usage and Tasks
### Tasks
- Predict reading comprehension - [description](descriptions/models/predict.md)
- Cluster eye tracking and reading comprehension data -[description](descriptions/models/cluster.md)

### Usage
Run `python` + 
- `training\classifier_trial.py` - [MECO] train a classifier to predict the reading comprehension score for a single participant's trials (passage level)
- `training\clustering_trial.py` - [MECO] identify clusters of trial data (passage level)
- `visualizations\saccade_plot.py` - [MECO] plot saccades for a participant's trial
 