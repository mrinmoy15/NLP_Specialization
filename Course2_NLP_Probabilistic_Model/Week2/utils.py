import string
from collections import defaultdict


# Punctuation characters
punct = set(string.punctuation)

# Suffixes
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def assign_unkown(word):
    """
    This function assign a token to an unknown word.
    
    Parameters:
    ------------
    word: string
        word to be assigned if unkown
    
    Returns:
    --------
    token: string
        token it is assigned to
    """

     # Loop the characters in the word, check if any is a digit
    if any(char.isdigit() for char in word):
        token = '--unk_digit--'
    
    # Loop the characters in the word, check if any is a punctuation character
    elif any(char in punct for char in word):
        token = '--unk_punct--'
    
    # Loop the characters in the word, check if any is an upper case character
    elif any(char.isupper() for char in word):
        token = '--unk_upper--'
    
    # Check if word ends with any noun suffix
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        token = '--unk_noun--'
    
    # Check if word ends with any verb suffix
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        token = '--unk_verb--'
    
    # Check if word ends with any adjective suffix
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        token = '--unk_adj--'
    
    # Check if word ends with any adverb suffix
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        token = '--unk_adv--'
    
    else:
        token = '--unk--'
    
    return token




def get_word_tag(line, vocab):
    """
    This function returns a word and tag pair for a given input line

    Parameters:
    -----------
    line: string
        input line
    vocab: list
        vocabulary list
    
    Returns:
    --------
    a (word, tag) tuple

    """

    # If line is empty return placeholders for word and tag
    if not line.split():
        word = '--n--'
        tag = '--s--'
    
    else:
        # Split line to separate word and tag
        word, tag = line.split()

        # Check if word is not in vocabulary
        if word not in vocab: 
            # Handle unknown word
            tag = assign_unkown(word)
    
    return word, tag



def preprocess(vocab, data_fp):
    """
    Preprocess data
    """

    orig = []
    prep = []
    
    # Read data
    with open(data_fp, 'r') as datafile:

        for cnt, word in enumerate(datafile):

            # End of sentence
            if not word.split():
                orig.append(word.strip())
                word = '--n--'
                prep.append(word)
                continue
            
            # Handle unknown words
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unkown(word)
                prep.append(word)
                continue

            else:
                orig.append(word.strip())
                prep.append(word.strip())

    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep



def create_dictionaries(training_corpus, vocab, verbose = True):
    """
    This function creates three dictionaries containg transition counts, emission counts and tag counts

    Parameters:
    ------------
    training corpus: list
        a corpus where each line has a word followed by its tag
    vocab: dictionary
        a dictionary where keys are words in vocabulary and value is an index
    verbose: boolean (default True)
        print the word counts if True
    
    Returns:
    ---------
        emission_counts, transition_counts, tag_counts
    """

    # initialize the dictionaries using defaultdict
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'
    prev_tag = '--s--'

    # use 'i' to track the line number in the corpus
    i = 0

    for word_tag in training_corpus:
        
        # Increment the word_tag count
        i += 1

        # Every 50,000 words, print the word count
        if i % 50000 == 0 and verbose:
            print(f"word count = {i}")
        
        # get the word and tag using the get_word_tag helper function
        word, tag = get_word_tag(word_tag, vocab)

        # Increment the transition count for the previous word and tag
        transition_counts[(prev_tag, tag)] += 1

        # Increment the emission count for the tag and word
        emission_counts[(tag, word)] += 1

        # Increment the tag count
        tag_counts[tag] += 1

        # Set the previous tag to this tag (for the next iteration of the loop)
        prev_tag = tag

    return emission_counts, transition_counts, tag_counts






