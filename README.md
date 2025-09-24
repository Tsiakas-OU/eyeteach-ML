# [EYE-TEACH] Machine Learning models
![Static Badge](https://img.shields.io/badge/status-work_in_progress-orange) 

**🚧 Under Active Development | Draft Version | Pre-Release** 

Development of ML models using eye-tracking data for reading comprehension tasks 

## Attribution
For the purpose of the EYE-TEACH project, we sourced existing datasets and codes from:
- [MECO-L1](https://osf.io/3527a/), under the [CC-By Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license 
- [WebQAmGaze](https://github.com/tfnribeiro/WebQAmGaze), under the [CC-By Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license  

## Installation
see `requirements.txt` or `environment.yml`
- Using `environment.yml` for conda environment:  
    - `conda env create -f environment.yml` 
    - use `conda activate sklearn` to activate environment
- Using `requirements.txt` to install dependencies in a new environment: 
    - `pip install -r requirements.txt`

## Folders
- `datasets/`: training datasets for model training
- `visualizations/`: codes for data visualizations
- `explanations/`: codes for shap values and LLM-based explanations
- `training/`: codes for model training
- `trained_models/`: trained models

## Methods - Visualizations and Explanations
- plot saccades path for a single participant's trial - MECO L1 - (`visualizations\saccade_plot_meco.py`)
- plot fixation path for a single participant's trial - MECO L1 - (`visualizations\fixation_plot_meco.py`)
- (Draft) visualization of LLM-based cluster analysis - open `visualizations\cluster_analysis_example.html`
- SHAP explanations - global and local explanations based on SHAP values (`explanations\shap_explanations.py`)
- LLM feedback - communicate model parameters and explanations (`explanations\LLM_explanations.py`)

## Usage and Tasks
### Tasks -- see [description](training/tasks.md)
- Predict reading comprehension
- Cluster eye tracking and reading comprehension data

### Usage
Run `python` + 
- `training\classifier_trial_meco.py` - [MECO] train a classifier to predict the reading comprehension score for a single participant's trials (passage level)
- `training\clustering_trial_meco.py` - [MECO] identify clusters of trial data (passage level)
- `visualizations\saccade_plot_meco.py` - [MECO] plot saccades for a participant's trial

## License
External components are subject to their respective licenses.
