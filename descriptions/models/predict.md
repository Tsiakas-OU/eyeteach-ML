# predict reading comprehension

## tasks
1. predict reading comprehension for a single participant's trials - [code](../../classifier_trial.py): 
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