# R codes to extract MECO-L1 datasets -- CURRENTLY ONLY FOR EN_UK
- version used: `release 2.0/version 2.0/wave 2`
- you first need to download the data from `https://osf.io/3527a/files`
- place the *.R codes in the `\code` folder and run the codes

## load_trial-ET.R
extracts the passage-level data `(primary/eye tracking data/joint_passage_trimmed_wave2_version2.0.RDA)` and joins the `ACCURACY` scores from `(primary/comprehension data)`
- file: `MECO_en_uk-passage.csv`

## load_fix_sac_reports.R
extract the fixation `(primary/eye tracking data/joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA)` and saccade reports 
`(primary/eye tracking data/joint_sac_trimmed_l1_wave2_version2.0.RDA)` and joins the `ACCURACY` scores from `(primary/comprehension data)`
- files: `MECO-fix_data_en_uk.csv` and `MECO-sac_data_en_uk.csv`
