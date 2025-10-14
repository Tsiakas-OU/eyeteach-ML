rm(list = ls())

library(ggplot2)
library(tidyverse)
library(psych)
library(xtable)
library(cowplot)
library(reshape2)
library(Hmisc)


# Function to pause and wait for user input
wait_for_input <- function() {
  invisible(readline(prompt = "Press [Enter] to continue or [Esc] to stop..."))
}

# load eye tracking and comprehension data - passage ----------------
load("../primary data/eye tracking data/joint_passage_trimmed_wave2_version2.0.rda")
load("../primary data/comprehension data/joint_l1_Wave2_acc_breakdown_trimmed.rda")

# load en_uk data (single language)
passage_data.en_uk <- joint.passage[joint.passage$lang == "en_uk", ]
comp_data.en_uk <- joint.br.acc[joint.br.acc$lang == "en_uk", ]


# merge eye tracking with comprehension data ------------------------------
# First, make sure both data frames have the uniform_id column
if (!("uniform_id" %in% names(comp_data.en_uk)) || !("uniform_id" %in% names(passage_data.en_uk))) {
  stop("Both data frames must have a 'uniform_id' column")
}

# Get unique participant IDs present in both datasets
participants <- intersect(comp_data.en_uk$uniform_id, passage_data.en_uk$uniform_id)
cat("Found", length(participants), "participants with data in both frames\n\n")

# Initialize empty list to store merged data for each participant
merged_list <- list()
missing_trialids <- character()

# Variables to keep in final output
final_vars <- c("uniform_id", "trialid", "trial", "trial.nwords", "nblink", "nrun", "nfix", 
                "nout", "sac", "skip", "refix", "reg", "mfix", "firstpass", 
                "rereading", "total", "rate", "ACCURACY")

# Process each participant
for (id in participants) {
  # Get passage data for this participant
  passage_subset <- passage_data.en_uk[passage_data.en_uk$uniform_id == id, ]
  
  # Get comp data for this participant
  comp_subset <- comp_data.en_uk[comp_data.en_uk$uniform_id == id, ]
  
  # Initialize temporary merged data for this participant
  participant_merged <- data.frame()
  
  # Process each trial in comp data
  for (i in 1:nrow(comp_subset)) {
    trial_number <- comp_subset$number[i]
    matching_passage <- passage_subset[passage_subset$trialid == trial_number, ]
    
    if (nrow(matching_passage) == 0) {
      missing_trialids <- c(missing_trialids, paste(id, trial_number, sep = ":"))
      next  # Skip to next trial if no passage data
    }
    
    # Merge comp and passage data for this trial
    merged_trial <- cbind(
      comp_subset[i, c("uniform_id", "ACCURACY")], 
      matching_passage[, c("trialid", "trial", "trial.nwords", "nblink", "nrun", "nfix", 
                           "nout", "sac", "skip", "refix", "reg", "mfix", 
                           "firstpass", "rereading", "total", "rate")]
    )
    
    participant_merged <- rbind(participant_merged, merged_trial)
  }
  
  # Add to main list if we found any matches
  if (nrow(participant_merged) > 0) {
    merged_list[[as.character(id)]] <- participant_merged
  }
}

# Combine all participants' data
final_merged <- do.call(rbind, merged_list)

# Report missing trialids
if (length(missing_trialids) > 0) {
  cat("\nMissing passage data for", length(missing_trialids), "trialids:\n")
  print(missing_trialids)
} else {
  cat("\nAll trialids had matching passage data\n")
}

# Ensure we have all requested variables
final_merged <- final_merged[, final_vars]

# Check results
cat("\nFinal merged data contains", nrow(final_merged), "rows\n")
str(final_merged)
head(final_merged)

output_filename <- "MECO-en_uk-passage.csv"

# Save to CSV with proper formatting
write.csv(final_merged, 
          file = output_filename,
          row.names = FALSE,  # Don't save row numbers
          na = "",           # Represent missing values as empty strings
          fileEncoding = "UTF-8")  # Use UTF-8 encoding for special characters

# Verify the file was created
if (file.exists(output_filename)) {
  cat("\nSuccessfully saved merged data to:", output_filename, "\n")
  cat("File contains", nrow(final_merged), "rows and", ncol(final_merged), "columns\n")
} else {
  warning("Failed to save the CSV file")
}