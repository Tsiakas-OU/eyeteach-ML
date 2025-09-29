# clustering reading comprehension data
## clustering of trial data for all participants/trials - `training\clustering_trial_meco.py`:  
    - input (features):
        - 'nblink'
        - 'nrun'
        - 'nfix'
        - 'nout' 
        - 'sac'
        - 'skip' 
        - 'refix'
        - 'reg'
        - 'mfix' 
        - 'firstpass'
        - 'rereading'
        - 'total' 
        - 'rate'
        - 'ACCURACY'
    - output: cluster summary 

# predict reading comprehension
## predict reading comprehension for the trials of a new participant - `training\classifier_participant_meco.py`: 
- input (features): 
    - `nblink`
    - `nrun`
    - `nfix`
    - `nout` 
    - `sac`
    - `skip` 
    - `refix`
    - `reg`
    - `mfix` 
    - `firstpass`
    - `rereading`
    - `total` 
    - `rate`
- output: `ACCURACY_CLASS` - 3 classes (low_medium, high, perfect)

## predict reading comprehension for a new trial (text) from all participants - `training\classifier_trial_meco.py`: 
- input (features): 
    - `nblink`
    - `nrun`
    - `nfix`
    - `nout` 
    - `sac`
    - `skip` 
    - `refix`
    - `reg`
    - `mfix` 
    - `firstpass`
    - `rereading`
    - `total` 
    - `rate`
- output: `ACCURACY_CLASS` - 3 classes (low_medium, high, perfect)

#### note: ACCURACY [0-4] is discretized into: 
- [0-2] - low_medium
- [3] - high
- [4] - perfect 
