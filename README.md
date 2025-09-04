# [EYE-TEACH] Machine Learning models
development of ML models using eye-tracking data for reading comprehension tasks

## Installation
- requirements - see `requirements.txt` or `environment.yml` for virtual/conda environments
- if needed, create/activate virtual environment (e.g., conda)

## Data Used
- `MECO-en_uk-passage.csv`: MECO-L1 passage data for English (UK). It includes ET features and reading comprehension scores [0-4] for each passage/trial [1-12] - [see more](descriptions/data/mecoL1.md)
- 

## Methods ()
- visualizations - 
- SHAP - global and local explanations (`shap_explanations.py`)
- LLM - communicate model explanations (`LLM_explanations.py`)

## Usage and Tasks
- Tasks
    - Predict reading comprehension - [description](descriptions/models/predict.md)
    - Cluster eye tracking and reading comprehension data -[description](descriptions/models/cluster.md)

- Run `python` + 
    - `classifier_trial.py` -- predict the reading comprehension score for a single trial
    - `classifier_participant.py` -- predict the reading comprehension score for a participant
    - `clustering_trial.py` -- identify clusters of trial data
