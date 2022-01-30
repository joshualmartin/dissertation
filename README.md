# Dissertation Pipelines

This set of code and data is a result of the dissertation work titled: Automatic Speech Recognition Systems, Spoken Corpora, and African American Language: An Examination of Linguistic Bias And Morphosyntactic Features. The Python code constructed here is primarily in Jupyter Notebooks files and will need Jupyter to run. The folders contain the data and data analyses performed using the code.

There are two larger sections of code contained within this repository: (1) Code written to parse and analyze spoken corpora data, and (2) Code written to test automatic speech recognition systems and analyze their outputs. Each section is divided into sub-sections labeled by step number in order to have a more manageable pipeline. The two sections and their sub-sections are detailed here.

## Section 1: Analyzing Spoken Corpora Data
### Step 1.1: Creating Dataframes from Corpora
Step 1.1 uses the Python package pandas to convert spoken corpora datasets into pandas dataframes for analysis.

### Step 1.2: Determining the Habituality of Be
Step 1.2 is a defined function focused on automatically determining the habitual nature of the verb "to be" in spoken corpora datasets.

### Step 1.3: Gold Standard csv Files
Step 1.3 is a folder of gold standard csv files used for the rest of the analysis.

### Step 1.4: Splitting Utterance Content
Step 1.4 splits corpora data into utterance level chunks for analysis.

### Step 1.5: Manually Annotated csv Files
Step 1.5 is a folder of manually annotated csv Files used for the rest of the analysis.

### Step 1.6: Getting Quantitative Information about Feature Usage
Step 1.6 analyzes the data and provides detailed quantitative information on the contents of the spoken corpora.

### Step 1.7: Getting Word Types per Structural Patterns
Step 1.7 provides detailed analyses of the word types cross-listed with structural patterns.

### Step 1.8: Getting n-Gram Dataframes
Step 1.8 produces dataframes which provide information on the n-gram structures surrounding the feature in question.

## Section 2: Analyzing Automatic Speech Recognition Systems
### Step 2.0: Creating Dataframes from Corpora for Ain't Variations
Step 2.0 produces dataframes from spoken corpora for variations of syntactic context analogous to ain't structures.

### Step 2.1: Parsing CORAAL Audio
Step 2.1 parses CORAAL audio files using the Python package parselmouth.

### Step 2.2: Getting Signal to Noise Ratio
Step 2.2 calculates Signal to Noise Ratio (SNR) in audio files.

### Step 2.3: Getting Speech Rate
Step 2.3 calculates speech rate for audio files.

### Step 2.4: Getting Transcriptions from Automatic Speech Recognition (ASR) Services
Step 2.4 accesses ASR services and transcribes audio files.

### Step 2.5: Cleaning Utterance Content
Step 2.5 standardizes transcription idiosyncracies for valid comparison between gold standard (manually performed) transcriptions and ASR transcriptions.

### Step 2.6: Getting Word Counts
Step 2.6 calculates word counts for transcriptions.

### Step 2.7: Getting Word Error Rates (WER)
Step 2.7 calculates word error rates for ASR transcriptions.

### Step 2.8: Getting Error Counts
Step 2.8 calculates the number of errors in ASR transcriptions.

### Step 2.9: Getting Word Error Rates (WER) Pre- and Post-Feature
Step 2.9 calculates word error rates before and after the feature in question.

### Step 2.10: Checking ASR Outputs for the Feature
Step 2.10 checks ASR transcriptions for the feature in question.

### Step 2.11: Checking Adjacent Tokens
Step 2.11 checks the adjacent tokens to the feature in question.

### Step 2.12: Creating the Correctness Column
Step 2.12 creates a column in the dataframe for transcript correctness.

### Step 2.12.5: Getting Habituality and Completive Columns
Step 2.12.15 automatically determines the habitual (i.e. invariant) nature of "to be" in a transcript and the completive (i.e. perfective) nature of "done" in a transcript.

### Step 2.13: Criteria for Manually Judging Correctness
Step 2.13 provides the criteria used to manually judge correctness for "to be" and "done".

### Step 2.14: Getting Descriptive Statistics
Step 2.14 provides descriptive statistical results from the data.

### Step 2.15: Final Statistics
Step 2.15 provides R code which calculates the final statistics in the analysis.
