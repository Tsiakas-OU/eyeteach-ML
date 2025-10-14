# Load required libraries
library(dplyr)

# load EN_UK fixation/saccade report and comprehension data - passage ----------------
load("../primary data/eye tracking data/joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA")
load("../primary data/eye tracking data/joint_sac_trimmed_l1_wave2_version2.0.RDA")
load("../primary data/comprehension data/joint_l1_Wave2_acc_breakdown_trimmed.rda")

# Create sac_data_en_uk.csv
# Convert 'trialid' in joint.br.acc to numeric before joining
sac_data_en_uk <- joint.sac %>%
  filter(lang == "en_uk") %>%
  left_join(
    joint.br.acc %>%
      filter(lang == "en_uk") %>%
      select(uniform_id, number, ACCURACY) %>%
      rename(trialid = number) %>%
      mutate(trialid = as.numeric(trialid)), # Convert the key to numeric
    by = c("uniform_id", "trialid")
  )

# Create fix_data_en_uk.csv
# Convert 'trialid' in joint.br.acc to numeric before joining
fix_data_en_uk <- joint.fix %>%
  filter(lang == "en_uk") %>%
  left_join(
    joint.br.acc %>%
      filter(lang == "en_uk") %>%
      select(uniform_id, number, ACCURACY) %>%
      rename(trialid = number) %>%
      mutate(trialid = as.numeric(trialid)), # Convert the key to numeric
    by = c("uniform_id", "trialid")
  )

# Save the files
write.csv(sac_data_en_uk, "sac_data_en_uk.csv", row.names = FALSE)
write.csv(fix_data_en_uk, "fix_data_en_uk.csv", row.names = FALSE)

cat("Files saved successfully!\n")