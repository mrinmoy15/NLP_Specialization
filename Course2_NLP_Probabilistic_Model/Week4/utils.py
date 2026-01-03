import numpy as np
from scipy import linalg
from collections import defaultdict
import re
import emoji
import nltk


def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def get_idx(words, word2Ind):
    """
    This function returns the indicies of a given list of words.

    Parameters:
    -----------
    words: list of strings
        list of input words
    word2Ind: dictionary
        Dictionary containing words and corresponding indices as key value pairs

    Returns:
    --------
    idx: list of integers
        List of indicies of a given list of words
    """
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx




def pack_idx_with_frequency(context_words, word2Ind):
    """
    This functions creates the a list of index and frequency of occurance of a context words list

    Parameters:
    -----------
    context_words: list of strings
        list of context words
    word2Ind: dictionary
        Dictionary containing word and index as key value pairs
    
    Returns:
    ---------
    packed: list of tuples
        list of tuples containing index and frequency of the context words 
    
    """
    freq_dict = defaultdict(int)
    
    for word in context_words:
        freq_dict[word] += 1
    
    idxs = get_idx(context_words, word2Ind)

    packed = []

    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    
    return packed
    



def get_vectors(data, word2Ind, V, C):
    """
    This functions creates training set vectors for a given dataset

    Parameters:
    -----------
    data: list of strings
        List of input words
    word2Ind: dictionary
        Dictionary containing word and its index as key value pair
    V: int
        Vocabulary size
    C: int
        half contact size
    
    Yields:
    -------
    x, y: tuple of arrays

    """
    i = C
    while True:
        
        y = np.zeros(V)
        x = np.zeros(V)
        
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        
        context_words = data[(i - C) : i] + data[(i + 1) : (i + C + 1)]
        num_ctx_words = len(context_words)
        
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq / num_ctx_words
        
        yield x, y

        i += 1
        if i >= len(data):
            print("i is being set to 0")
            i = 0


def get_batches(data, word2Ind, V, C, batch_size):
    """ 
    This function generates training or testing batches

    Parameters:
    -----------
    data: list of strings
        list of words
    word2Ind: dictionary
        Dictionary containing word and its index as key value pair
    V: int
        Vocabulary size
    C: int
        half contact size
    batch_size: int
        batch size
    
    Yields:
    --------
        batch_x, batch_y : numpy array
    """
    batch_x = []
    batch_y = []

    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []



def get_dict(data):
    """
    Input:
        K: the number of negative samples
        data: the data you want to pull from
        indices: a list of word indices
    Output:
        word_dict: a dictionary with the weighted probabilities of each word
        word2Ind: returns dictionary mapping the word to its index
        Ind2Word: returns dictionary mapping the index to its word
    """
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word



def compute_pca(data, n_components=2):
    """
    Input: 
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output: 
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    ### START CODE HERE ###
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    ### END CODE HERE ###
    return np.dot(evecs.T, data.T).T



# Define the 'get_windows' function
def get_windows(words, C):
    """
    This function produces pairs of context words and center word traversing through the sliding window of the input words(tokens)
    
    Parameters:
    ------------
    words: list
        A list of words (or tokens)
    C: int
        The context half-size, for a given center word, the context words are made of C words to the left and C words to 
        the right of the center word
    
    Yields:
    ---------
    (context_words, center_word) : (list, string)
        A tuple containg the context words and the center word

    """
    i = C

    while(i < len(words) - C):
        center_word = words[i]
        context_words = words[i-C : i] + words[i+1 : i + C +1]
        yield context_words, center_word
        i += 1




def word_to_one_hot_vector(word, word2Ind, V):
    """ 
    This function creates a one hot vector representation for a given word

    Parameters:
    -----------
    word: string
        input word
    word2Ind: dictionary
        dictionary containing (word:index) key value pairs
    V: int
        vocabulary size
        
    Returns:
    --------
    one_hot_vector: numpy array
        one hot vector representation of a given word

    """
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector



# Define the 'context_words_to_vector' function
def context_words_to_vector(context_words, word2Ind, V):
    """ 
    This function creates a vector representation of a given context words.

    Parameters:
    -----------
    context_words: list of strings
        A list of context words
    word2Ind: dictionary
        A dictionary containing (word, Index) key value pairs
    V: int
        Vocabulary size
    
    Returns:
    --------
    context_words_vectors: numpy array
        An array representing the given context words
    """
    context_words_vectors = [ word_to_one_hot_vector(word, word2Ind, V) for word in context_words ]
    context_words_vectors = np.mean(context_words_vectors, axis = 0)
    return context_words_vectors




# Define the generator function 'get_training_example'
def get_training_example(words, C, word2Ind, V):
    """ 
    This function generates training data for CBOW model given words(tokens)

    Parameters:
    ------------
    words: list of strings
        List containing the word tokens
    C: int
        half context size
    word2Ind: dictionary
        Dictionary containing the word and its index as key value pairs
    V: int
        Vocabulary size
    
    Yields:
    -------
    context_words_vector, center word vector: numpy arrays
    """
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)



def relu(x):
    """ 
    This functions gives ReLU output for an input array

    Parameters:
    ------------
    x: numpy array
        input array
    
    Returns:
    --------
    result: numpy array
        ReLU activation applied to the input array
    """
    result = x.copy()
    result[result < 0] = 0
    return result



def softmax(z):
    """ 
    This function gives softmax output of an input array

    Parameters:
    -----------
    z: numpy array
        input array
    
    Returns:
    --------
    softmax activation applied to an input array
    """
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z, axix = 0)
    return e_z / sum_e_z




def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = '(' + '|'.join(re.escape(u) for u in emojis) + ')'
    return re.compile(pattern)

exp = get_emoji_regexp()
print(exp.sub(repl='[emoji]', string='A 🏌️‍♀️ is eating a 🥐'))



def tokenize(corpus):
    """
    This function tokenize the input corpus for the Continous Bag Of Words (CBOW) model.
    
    Parameters:
    ------------
    corpus:string
        input corpus
    
    Returns:
    --------
    data: list
        list of tokens extracted from the input corpus
    
    """
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data)
    data = [ch.lower() for ch in data if ch.isalpha() or ch == '.' or get_emoji_regexp().search(ch)]
    return data