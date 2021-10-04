def determine_be_habituality(be_instances_df):
    
    """
    This code is only for the AAL morphosyntactic (grammatical) feature
    habitual 'be' (also known as aspectual 'be' and invariant 'be')
    see this link for info about habitual 'be': https://ygdp.yale.edu/phenomena/invariant-be

    This code will take the instances dataframes from the previous step 
    in the pipeline "creating_dataframes_from_corpora.py" and produce
    a dataframe that automatically determines whether the 'be' in the 
    row's Content is non-habitual. It labels non-habituality using 
    established linguistic patterns. Those that are not labeled as non-habitual
    either (1) contain more than one instance of 'be', (2) occur at the beginning
    of the utterance with no preceding word tokens, (3) are habitual 'be',
    (4) aren't captured by the non-habitual rules for some other reason.
    The produced dataframe will need to be manually inspected and corrected
    for habitual 'be's. The non-habitual labels should not need inspecting.
    The code will produce a csv file for manual inspection and correction. 
    """

    import string
    import pandas as pd
    import numpy as np
    from nltk import word_tokenize


    #In English, if 'be' is preceded (or governed in linguistics terms) by a modal or a negated
    # modal, it is non-habitual without exception. So, we can search for the word preceding 'be'
    # in the Content line and if it is a modal, then we can automatically determine it is non-habitual.
    # L1 means one word to the left of 'be'.
    modals_L1 = ["can", "can't", "cannot", "could", "couldn't", "may", "might", "must", "mustn't", "should", 
               "shouldn't", "ought", "hafta", "oughta", "will", "won't", "would", "wouldn't", "shall",
               "'ll", "'d", "ll", "d", "twill", "wouldst", "would'st", "shalt", "wilt", "twould", 
               "mayst", "wouldnt", "neednt", "mus", "wud", "shouldst", "to", "ter"] 
               #'to'/"ter" is not a modal, but it's included here for use with the L2 combo list

    #These are other various words that only precede a non-habitual 'be'.
    # # L1 means one word to the left of 'be'.
    others_L1 = ["had", "tryna", "gonna", "going", "sposta", "supposed", "finna", "gotta", "wanna", 
               "lemme", "to", "need", "needn't", "rather", "i'm'a", "let's", "lets", 
               "liketa", "better", "got", "want", "wanna", "please", "na", "ta", "than",
               "if", "let", "whether", "letting", "uh", "though", "them", "us", "me", "her", "him",
               "best", "ud", "blessed", "praised", "glory", "praise", "ull", "'m", "hadda", "why"]

    #Adverbs, often adverbs of frequency, can intervene between a modal and 'be'; however, even though
    # 'be' does not follow the modal directly in this case, it is still syntactically governed by the 
    # modal and thus will be non-habitual without exception. This is a list of potential adverbs (and other words)
    # that occur between the modal and 'be'.
    adverbs_and_others = ['again', 'all', 'almost', 'also', 'always', 'actually', 'annually', 'constantly', 'daily', 
             'eventually', 'even', 'ever', 'frequently', 'generally', 'hourly', 'infrequently',
             'just', 'later', 'like', 'monthly', 'never', 'next', 'nightly', 'normally', 'not',
             'now', 'occasionally', 'often', 'only', 'periodically', 'possibly', 'probably', 'quarterly', 
             'rarely', 'really', 'regularly', 'seldomly', 'sometimes', 'sometime', 'soon', 'still', 'then', 
             'today', 'tonight', 'very', 'weekly', 'well', 'yearly', 'yesterday', 'yet', 'that']

    #creates a combined list of the two previous lists
    full_L1 = modals_L1 + others_L1 + adverbs_and_others

    #If these punctuation markers occur immediately to the left of 'be', then it is non-habitual
    # most likely, it is in the imperative mood.
    # L1 means one word to the left of 'be'.
    punctuation_L1 = ",.!?"

    #list of coordinating conjunctions. if 'be' is the second word token in the utterance, and
    # a coordinating conjunction is the first, it will be labeled non-habitual as this analysis
    # does not take into consideration larger context of speaker turn
    coordinating_conjunctions = ['and', 'but', 'so', 'for', 'yet', 'or', 'nor']

    #creates a list of lists. Each list is pair of a modal and an adverb
    # L2 means two words to the left of 'be'
    L1s_adverbs_L2 = [[L1, adverb] for L1 in full_L1 for adverb in adverbs_and_others]

    #creates a list of pronouns to be used in collocations with modals
    # note, that this should only be used with modals since PRONOUN + HABITUAL BE is a pattern
    # that occurs in actual speech
    pronouns = ["I", "you", "he", "she", "it", "you", "we", "they", "thou"]

    #creates a list of collocations with modals first. this is for questions (i.e., interrogative) structure
    modals_pronouns_L2 = [[modal, pronoun] for modal in modals_L1 for pronoun in pronouns]

    #other various two-word combination (collocation) that could occur before
    # a non-habitual 'be'. L2 means two words to the left of 'be'
    others_L2 = [["gon", "na"], ["got", "ta"], ["had", "better"], ["'m", "a"], ["wan", "na"], ["can", "n't"], ["can", "not"],
    ["could","n't"], ["must", "n't"], ["should", "n't"], ["wo", "n't"], ["would", "n't"], ["'", "a"], ["let", "'s"], ["i", "'m"]]

    #combines the previous two lists
    full_L2_collocations = L1s_adverbs_L2 + others_L2 + modals_pronouns_L2

    #these are words that occur two or three words to the left of be that will make it non-habitual
    # regardless of what intervenes
    other_L2_L3_single = ["if", "whether", "let", "letting"]
    full_L2_single = other_L2_L3_single + modals_L1

    #if 'be' is the third word token in the utterance, an adverb is the second, and
    # a coordinating conjunction is the first, it will be labeled non-habitual as this analysis
    # does not take into consideration larger context of speaker turn
    conjunctions_adverbs_L2 = [[conjunction, adverb] for conjunction in coordinating_conjunctions for adverb in adverbs_and_others]

    #creates a list of L1-L3 collocations with a modal then pronoun then adverb
    modal_pronoun_adverb_L3 = [[modal, pronoun, adverb] for modal in modals_L1 for pronoun in pronouns for adverb in adverbs_and_others]

    #creates a list L1-L3 collocations with the full_L1 list and two adverbs intervening
    L3_adverb_adverb = [[L1, adverb1, adverb2] for L1 in full_L1 for adverb1 in adverbs_and_others for adverb2 in adverbs_and_others]

    #creates a full list of L1-L3 collocations
    full_L3_collocations = modal_pronoun_adverb_L3 + L3_adverb_adverb

    #creates an empty column in the dataframe to append the habituality of the 'be' in the line to
    # the following code will append a number depending on the habituality of the 'be'
    # this analysis will only be within the context of the utterance;
    # therefore, if 'be' is the first or last word token in the utterance,
    # it is labeled non-habitual regardless of larger speaker turn context
    # in previous versions of this, I took the larger into consideration, but decided
    # against it because the ASR functionality analysis in later steps which will
    # take only the audio for utterances into account would be inconsistent
    # however, I have left the code key for what I did previously which was to label
    # when a 'be' was the first or last word token in the utterance in the multiple
    # hashed comments here. if you would like to use those, just change the code below
    # here is the key for the numbers:
    #### -2 = the 'be' occurs as the first word in the utterance and must be inspected manually
    #### -1 = the 'be' occurs as the last word in the utterance and must be inspected manually
    # 0 = no habitual 'be' present in the line
    # 1 or more = the count of potential habitual 'be' present in the line, this may not be correct and must be inspected manually
    be_instances_df["FeatureCountPerLine"] = np.nan

    #loops through the rows in the be_instances_df dataframe provided in the function
    for file_row in be_instances_df.itertuples():

      #if the number of 'be' is 1, then the row Content is cleaned
        if file_row.InstancesCountPerLine == 1:
            
          # tokenizes the words in the Content using nltk's word_tokenizer and lowercases the words
            content_words_tokenized_cleaned = [word.lower() for word in word_tokenize(file_row.Content)]

          ##########PREVIOUS VERSION#####################################################
          ###this code will do the following; however, I decided against it so that
          ### I could automatically rule out instances of 'be' that are preceded by 
          ### punctuation such as . or , since those will be non-habitual (usually
          ### imperative mood). Also, for this analysis, I have decided that anytime
          ### a 'be' instance is 'be-' or some sort of other punctuation, then it 
          ### will automatically be labeled non-habitual since most of the time these
          ### are instances of words stopped short, e.g. a speaker beginning to say
          ### the word "best" but only getting "be-" out. 
          #(1) lowercases the words, (2) strips punctuation that occurs to the right
          # of words. This allows for apostrophes to the left of separated 
          # contractions to remain, (3) removes tokens that are only punctuation
          #  content_words_tokenized_cleaned = [word.lower().rstrip(string.punctuation) 
          #                                     for word in content_words_tokenized 
          #                                     if word not in string.punctuation]
          ###############################################################################

          #loops through the cleaned and tokenized content words
            for content_word in content_words_tokenized_cleaned:

              #skips all other words but 'be'
                if content_word in ["be", "Be"]:

                  #gets the index of 'be' in the list
                    be_index = content_words_tokenized_cleaned.index(content_word)

                  #if the index is 0, meaning 'be' is the first token in the utterance,
                  # appends a 0 and labels as non-habitual. if you would like to differentiate
                  # first word token occurences from other 0 values, simply change the 
                  # assigned value here (see code key above)
                    if be_index == 0:
                        be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                  #if 'be' is the last token in the utterance,
                  # appends a 0 and labels as non-habitual. if you would like to differentiate
                  # first word token occurences from other 0 values, simply change the 
                  # assigned value here (see code key above)
                    elif content_word == content_words_tokenized_cleaned[-1]:
                        be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                  #if 'be' is the second word token in the utterance, and
                  # a coordinating conjunction is the first, it will be labeled non-habitual as 
                  # this analysis does not take into consideration larger context of speaker turn
                    elif be_index == 1 and content_words_tokenized_cleaned[0] in coordinating_conjunctions:
                        be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                  #if 'be' is the third word token in the utterance, an adverbs is the second, and
                  # a coordinating conjunction is the first, it will be labeled non-habitual as 
                  # this analysis does not take into consideration larger context of speaker turn
                    elif be_index == 2 and content_words_tokenized_cleaned[:2] in conjunctions_adverbs_L2:
                        be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                  #habitual 'be' can be preceded by "don't", but not any other 
                  # negated contraction. This snippet will see if the preceding two 
                  # word tokens are "do", "n't" and appends a 1 if so (meaning habitual, 
                  # see code key above). Any other contracted negative (e.g., "wo", "n't") 
                  # will result in a 0 (meaning non-habitual, see code key above). 
                  # However, if "don't" is the first word in the utterance, that means
                  # 'be' is in the imperative mood and is being negated, which is non-habitual.
                  # All of this is necessary because the nltk tokenizer splits 
                  # contractions into two word tokens
                    elif content_words_tokenized_cleaned[be_index-1] == "n't":
                        if content_words_tokenized_cleaned[be_index-2] == "do":
                            if be_index-2 == 0:
                                be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0
                            else:
                                be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 1
                        else:
                            be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                  #checks the L2 to see if it's a contracted negative.
                  # if the L2 is "n't" and the L3 is "do" and it occurs as the first word token
                  # in the utterance, it's marked non-habitual. If it's not the first word token
                  # in the utterance, it's marked habitual in order to be checked manually.
                  # otherwise, if the L3 is not "do", it is marked non-habitual
                    elif content_words_tokenized_cleaned[be_index-2] == "n't":
                        if content_words_tokenized_cleaned[be_index-3] == "do":
                            if be_index-3 == 0:
                                be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0
                            else:
                                be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 1
                        else:
                            be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                  #checks the first word to the left and the first two words to the left
                  # of 'be' to see whether they appear in the L1 and L2 lists.
                  # if words have punctuation attached, it checks a version where punctuation is stripped
                  # on either side of the word
                  # if so, a 0 is appended (meaning non-habitual, see code key above)
                    elif (content_words_tokenized_cleaned[be_index-1] in full_L1 or
                      content_words_tokenized_cleaned[be_index-1].strip(string.punctuation) in full_L1 or
                      content_words_tokenized_cleaned[be_index-1] in punctuation_L1 or 
                      content_words_tokenized_cleaned[be_index-2:be_index] in full_L2_collocations or
                      [word.strip(string.punctuation) for word in content_words_tokenized_cleaned[be_index-2:be_index]] in full_L2_collocations or
                      content_words_tokenized_cleaned[be_index-2]in full_L2_single or
                      content_words_tokenized_cleaned[be_index-2].strip(string.punctuation) in full_L2_single or
                      content_words_tokenized_cleaned[be_index-3:be_index] in full_L3_collocations or
                      [word.strip(string.punctuation) for word in content_words_tokenized_cleaned[be_index-3:be_index]] in full_L3_collocations or
                      content_words_tokenized_cleaned[be_index-3] in other_L2_L3_single or
                      content_words_tokenized_cleaned[be_index-3] in modals_L1):
                          be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                  #any other case appends a 1 to the habituality list. This technically
                  # means habitual (see code key above). However, since all automatically
                  # determined habitual 'be' instances must be manually inspected,
                  # the 1 here acts as a catch-all to be manually inspected
                    else:
                        be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 1
                
                #marks instances of 'be' with puctuation attached, mainly "be-", as non-habitual
                elif (content_word in [f"be{punct}" for punct in string.punctuation] 
                    or content_word in [f"{punct}be" for punct in string.punctuation]):
                        be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 0

                #if content_word is not "be" or f"be{punct}", then restarts the loop
                else:
                    continue
        
      #if the number of 'be' is more than one in the row's Content
        elif file_row.InstancesCountPerLine > 1:
            #sets a count of habitual 'be's in the row to a foundation of zero
            # if at the end, the number remains 0, that means there are only non-habitual
            # 'be's in the row
            row_habitualBe_count = 0
            
            # tokenizes the words in the Content using nltk's word_tokenizer and lowercases the words
            content_words_tokenized_cleaned = [word.lower() for word in word_tokenize(file_row.Content)]
            
            #creates a list of the indices for all instances of 'be' in the row
            be_indices = [index for index, content_word in enumerate(content_words_tokenized_cleaned) if content_word == "be"]
            
            #loops through the be idices and performs the same evaluation as the code for the 
            # rows with only one instance of 'be'
            # however, instead of appending a 0 or 1 to the row's FeatureCountPerLine column
            # each condition does one of two things: (1) continues the code back to the 
            # start of the loop for the next row in the case of non-habitual 'be' or 
            # (2) increases the row_habitualBe_count by 1 for habitual 'be's or instances
            # missed by the non-habitual filters
            for be_index in be_indices:
                if be_index == 0:
                    continue

              #if 'be' is the second word token in the utterance, and
              # a coordinating conjunction is the first, it will be labeled non-habitual as 
              # this analysis does not take into consideration larger context of speaker turn
                elif be_index == 1 and content_words_tokenized_cleaned[0] in coordinating_conjunctions:
                    continue

              #if 'be' is the third word token in the utterance, an adverbs is the second, and
              # a coordinating conjunction is the first, it will be labeled non-habitual as 
              # this analysis does not take into consideration larger context of speaker turn
                elif be_index == 2 and content_words_tokenized_cleaned[:2] in conjunctions_adverbs_L2:
                    continue

              #habitual 'be' can be preceded by "don't", but not any other 
              # negated contraction. This snippet will see if the preceding two 
              # word tokens are "do", "n't" and appends a 1 if so (meaning habitual, 
              # see code key above). Any other contracted negative (e.g., "wo", "n't") 
              # will result in a 0 (meaning non-habitual, see code key above). 
              # However, if "don't" is the first word in the utterance, that means
              # 'be' is in the imperative mood and is being negated, which is non-habitual.
              # All of this is necessary because the nltk tokenizer splits 
              # contractions into two word tokens
                elif content_words_tokenized_cleaned[be_index-1] == "n't":
                    if content_words_tokenized_cleaned[be_index-2] == "do":
                        if be_index-2 == 0:
                            continue
                        else:
                            row_habitualBe_count += 1
                    else:
                        continue

              #checks the L2 to see if it's a contracted negative.
              # if the L2 is "n't" and the L3 is "do" and it occurs as the first word token
              # in the utterance, it's marked non-habitual. If it's not the first word token
              # in the utterance, it's marked habitual in order to be checked manually.
              # otherwise, if the L3 is not "do", it is marked non-habitual
                elif content_words_tokenized_cleaned[be_index-2] == "n't":
                    if content_words_tokenized_cleaned[be_index-3] == "do":
                        if be_index-3 == 0:
                            continue
                        else:
                            row_habitualBe_count += 1
                    else:
                        continue

              #checks the first word to the left and the first two words to the left
              # of 'be' to see whether they appear in the L1 and L2 lists.
              # if words have punctuation attached, it checks a version where punctuation is stripped
              # on either side of the word
              # if so, a 0 is appended (meaning non-habitual, see code key above)
                elif (content_words_tokenized_cleaned[be_index-1] in full_L1 or
                  content_words_tokenized_cleaned[be_index-1].strip(string.punctuation) in full_L1 or
                  content_words_tokenized_cleaned[be_index-1] in punctuation_L1 or 
                  content_words_tokenized_cleaned[be_index-2:be_index] in full_L2_collocations or
                  [word.strip(string.punctuation) for word in content_words_tokenized_cleaned[be_index-2:be_index]] in full_L2_collocations or
                  content_words_tokenized_cleaned[be_index-2]in full_L2_single or
                  content_words_tokenized_cleaned[be_index-2].strip(string.punctuation) in full_L2_single or
                  content_words_tokenized_cleaned[be_index-3:be_index] in full_L3_collocations or
                  [word.strip(string.punctuation) for word in content_words_tokenized_cleaned[be_index-3:be_index]] in full_L3_collocations or
                  content_words_tokenized_cleaned[be_index-3] in other_L2_L3_single or
                  content_words_tokenized_cleaned[be_index-3] in modals_L1):
                      continue

              #any other case appends a 1 to the habituality list. This technically
              # means habitual (see code key above). However, since all automatically
              # determined habitual 'be' instances must be manually inspected,
              # the 1 here acts as a catch-all to be manually inspected
                else:
                    row_habitualBe_count += 1
            
            #appends the number of habitual 'be' instances or 'be's that the filter missed
            # to the row in the dataframe
            be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = row_habitualBe_count
            
        else:
            be_instances_df.loc[file_row.Index, "FeatureCountPerLine"] = 1

    #returns the dataframe
    return be_instances_df


##the directory you want your dataframes to go to
## MAKE SURE IT ENDS WITH THE PROPER SLASH
# df_output_path = "/Users/benjaminlowe/Downloads/test_csv_results/"


##this will run the code and get you the habituality dataframes for each corpus
# coraal_habituality_df = determine_be_habituality(coraal_instances_df)
# switchboardHub5_habituality_df = determine_be_habituality(switchboardHub5_instances_df)
# fisher_habituality_df = determine_be_habituality(fisher_instances_df)
# librispeech_habituality_df = determine_be_habituality(librispeech_instances_df)
# timit_habituality_df = determine_be_habituality(timit_instances_df)

##this will change the order of the columns for each dataframe so they're
## more readable when you do the manual correction
# coraal_habituality_df = coraal_habituality_df[["File", "Line", "Speaker", "UttStartTime", "UttEndTime", "InstancesCountPerLine", "FeatureCountPerLine", "Content"]]
# switchboardHub5_habituality_df = switchboardHub5_habituality_df[["File", "Line", "Speaker", "UttStartTime", "UttEndTime", "InstancesCountPerLine", "FeatureCountPerLine", "Content"]]
# fisher_habituality_df = fisher_habituality_df[["File", "Line", "Speaker", "UttStartTime", "UttEndTime", "InstancesCountPerLine", "FeatureCountPerLine", "Content"]]
# librispeech_habituality_df = librispeech_habituality_df[["File", "Line", "InstancesCountPerLine", "FeatureCountPerLine", "Content"]]
# timit_habituality_df = timit_habituality_df[["File", "BeginningIntegerSampleNumber"", "EndIntegerSampleNumber", "InstancesCountPerLine", "FeatureCountPerLine", "Content"]]


##this will export the dataframes to csvs for you for manual correction
# coraal_habituality_df.to_csv(f"{df_output_path}coraal_habituality_df.csv")
# fisher_habituality_df.to_csv(f"{df_output_path}fisher_habituality_df.csv")
# switchboardHub5_habituality_df.to_csv(f"{df_output_path}switchboardHub5_habituality_df.csv")
# librispeech_habituality_df.to_csv(f"{df_output_path}librispeech_habituality_df.csv")
# timit_habituality_df.to_csv(f"{df_output_path}timit_habituality_df.csv")