# description of MECO-L1 datasets

- version used: `release 2.0/version 2.0/wave 2`
- check: https://github.com/sascha2schroeder/popEye/blob/master/materials/Measures.md

### MECO-en_uk-passage.csv
includes eye-tracking features, participant ID and reading comprehension scores for each trial

#### Features 
- `uniform_id`: Participant ID
- `trialid`: Number of trial
- `trial.nwords`: Number of words in trial
- `nblink`: Number of blinks in trial
- `nrun`: Number of runs on trial
- `nfix`: Number of fixations on trial
- `nout`: Number of outlier fixations on trial
- `sac`: Mean (forward) sacade length
- `skip`: Proportion of words in the trial that have been skipped during first-pass reading
- `refix`: Proportion of words in the trial that have been refixated
- `reg`: Proportion of words which have been regressed into
- `mfix`: Mean fixation duration
- `firstpass`: First-pass reading time (summed gaze duration for all words in a trial)
- `rereading`: Re-reading time (total reading time minus first-pass reading time)
- `total`: Total reading time
- `rate`: Reading rate (words per minute)
- `ACCURACY`: Number of correct responses - out of 4

### MECO-en_uk-comp.csv
Reading comprehension data for each participant's trial. Each trial (passage) has 4 comprehension questions.

#### Features 
- `uniform_id`: Participant ID
- `lang`: Language code 
- `trialid`: Number of trial
- `ACCURACY`: Number of correct responses - out of 4

### MECO-en_uk-sac.csv
Saccade report

#### Features
- `subid`: Participant ID (do not use - use `uniform_id` instead)
- `trialid`: Position of trial in analysis
- `trialnum`: Position of trial in experiment
- `itemid`: Item ID
- `cond`: Condition (if applicable)
- `sacid`: Number of saccade in a trial
- `msg`: type of saccade (saccade vs. blink)
- `xs`: Raw start x position (in pixel)
- `ys`: Raw start y position (in pixel)
- `xe`: Raw end x position (in pixel)
- `ye`: Raw end y position (in pixel)
- `xsn`: Corrected start x position (in pixel), i.e. after drift correction and line assignment
- `ysn`: Corrected start y position (in pixel), i.e. after drift correction and line assignment
- `xen`: Corrected end x position (in pixel), i.e. after drift correction and line assignment
- `yen`: Corrected end y position (in pixel), i.e. after drift correction and line assignment
- `start`: Start time (in ms since start of the trial)
- `stop`: End time (in ms since start of the trial)
- `dist.let`: Saccade length (in letters)
- `dur`: Duration (in ms)

### MECO-en_uk-fix_uk01.csv
Fixation report - for a single participant (uk01)

#### Features
- `subid`: Participant ID (do not use - use `uniform_id` instead)
- `trialid`: Position of trial in analysis
- `trialnum`: Position of trial in experiment
- `itemid`: Item ID
- `cond`: Condition (if applicable)
- `fixid`: Number of fixation in a trial
- `start`: Start time (in ms since start of the trial)
- `stop`: End time (in ms since start of the trial)
- `xs`: Raw x position (in pixel)
- `ys`: Raw y position (in pixel)
- `xn`: Corrected x position (in pixel), i.e. after drift correction and line assignment
- `yn`: Corrected y position (in pixel), i.e. after drift correction and line assignment
- `ym`: Mean y position (position of the line)
- `dur`: Duration
- `sac.in`: Incoming saccade length (in letters)
- `sac.out`: Outgoing saccade length (in letters)
- `type`: Whether fixation is an outlier fixation ("out"), i.e. located outside the text area (see assign.outlier and assign.outlier.dist arguments)
- `blink`: Whether a blink occured directly before or after the fixation
- `run`: Number of run the fixation was assigned to (if applicable)
- `linerun`: Number of run on the line the fixation was assigned to (if applicable)
- `line`: Number of line the fixation was assigned to
- `line.change`: Difference between the line of the current and the last fixation
- `line.let`: Number of letter on line
- `line.word`: Number of word on line
- `letternum`: Number of letter in trial
- `letter`: Name of Letter
- `wordnum`: Number of word in trial
- `word`: Name of Word
- `ianum`: Number of IA in trial
- `ia`: Name of IA
- `sentnum`: Number of sentence in trial
- `sent`: Name of sent (abbreviated)
- `sent.nwords`: Number of words in sentence
- `trial`: Name trial (abbreviated)
- `trial.nwords`: Number of words in trial
- `word.fix`: Number of fixation on word
- `word.run`: Number of run the word the word was read
- `word.runid`: Number of the word run, the fixation belongs to
- `word.run.fix`: Number of fixation within the run
- `word.firstskip`: Whether word has been skipped during first-pass reading
- `word.refix`: Whether word has been refixated with current fixation
- `word.launch`: Launch site distance from the beginning of the word
- `word.land`: Landing position with word
- `word.cland`: Centered landing position (e.g., calculated from the center of the word)
- `word.reg.out`: Whether a regression was made out of the word
- `word.reg.in`: Whether a regression was made into the word
- `ia.fix`: Number of fixation on IA
- `ia.run`: Number of run the word the IA was read
- `ia.runid`: Number of the IA run, the fixation belongs to
- `ia.run.fix`: Number of fixation within the run
- `ia.firstskip`: Whether IA has been skipped during first-pass reading
- `ia.refix`: Whether IA has been refixated with current fixation
- `ia.launch`: Launch site distance from the beginning of the IA
- `ia.land`: Landing position with IA
- `ia.cland`: Centered landing position (e.g., calculated from the center of the IA)
- `ia.reg.out`: Whether a regression was made out of the IA
- `ia.reg.in`: Whether a regression was made into the IA
- `sent.word`: Number of word in sentence
- `sent.fix`: Number of fixation on sentence
- `sent.run`: Number of run on sentence
- `sent.runid`: Number of the sentence run, the fixation belongs to
- `sent.firstskip`: Whether the sentence has been skipped during first-pass reading
- `sent.refix`: Whether sentence was refixated wither current fixation
- `sent.reg.out`: Whether a regression was made out the sentence
- `sent.reg.in`: Whether a regression was made into the sentence