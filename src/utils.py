import os
import pandas as pd

from nltk.tokenizer import WordPunctTokenizer
from collections import Counter

basepath = '../data'

def load_data():
    train = pd.read_csv(os.path.join(basepath, 'raw/train.csv'))
    test  = pd.read_csv(os.path.join(basepath, 'raw/test.csv'))

    return train, test

def load_sub():
    return pd.read_csv(os.path.join(basepath , 'raw/sample_submission.csv'))


def preprocess(train):
    """
    Create tokens and build an inverse token index
    a dictionary from token (string) to it's index in tokens (int)
    """
    tokenizer  = WordPunctTokenizer()
    
    trn_tokens = train['comment_text'].apply(lambda x: ' '.join(tokenizer.tokenize(str(x).lower())))
    trn_token_counts = Counter(' '.join(trn_tokens).split(' '))
    
    min_count = 10

    # tokens from token_counts keys that had at least min_count occurrences throughout the dataset
    tokens = list(filter(lambda x: trn_token_counts[x] > min_count, trn_token_counts.keys()))

    # Add a special tokens for unknown and empty words
    UNK, PAD = "UNK", "PAD"
    tokens   = [UNK, PAD] + tokens
    
    token_to_id = {}

    for tok in tokens:
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    
    return tokens, token_to_id

def as_matrix(sequences, token_to_id, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    UNK, PAD       = "UNK", "PAD"
    UNK_IX, PAD_IX = map(token_to_id.get, [UNK, PAD])

    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))
        
    max_len = min(max(map(len, sequences)), max_len or float('inf'))
    
    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
    for i,seq in enumerate(sequences):
        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix

def generate_batch(data, token_to_id, batch_size=None, replace=True, max_len=None, target_columns=[]):
    """
    Creates a pytorch-friendly dict from the batch data.
    :returns: a dict with {'title' : int64[batch, title_max_len]
    """
    if batch_size is not None:
        data = data.sample(batch_size, replace=replace)
    
    batch = {}
    batch['comment_text'] = as_matrix(data['comment_text'].values, token_to_id, max_len)
    batch['target']       = data.loc[:, target_columns].values
    
    return batch

