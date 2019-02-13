import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tnrange
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


from utils       import generate_batch, load_data, load_sub
from utils       import preprocess
from model_utils import GlobalMaxPooling, CommentsEncoder
from model_utils import compute_loss, compute_accuracy

SEED = 41
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

num_epochs = 2
max_len    = 200
batch_size = 32
batches_per_epoch = None


def iterate_minibatches(data, token_to_id, batch_size=32, max_len=None,
                        max_batches=None, shuffle=True, verbose=True, target_columns=[]):
    indices = np.arange(len(data))
    if shuffle:
        indices = np.random.permutation(indices)
    if max_batches is not None:
        indices = indices[: batch_size * max_batches]
        
    irange = tnrange if verbose else range
    
    for start in irange(0, len(indices), batch_size):
        yield generate_batch(data.iloc[indices[start : start + batch_size]],
                             token_to_id, 
                             max_len=max_len, target_columns=target_columns)


def train_with_cv(train):
    data_train, data_val = train_test_split(train, test_size=.3, random_state=SEED)
    tokens, token_to_id  = preprocess(data_train)
    
    target_columns = data_train.columns.tolist()[2:]

    UNK, PAD       = 'UNK', 'PAD'
    UNK_IX, PAD_IX = map(token_to_id.get, [UNK, PAD])

    comment_encoder = CommentsEncoder(out_size=64, n_tokens=len(tokens), PAD_IX=PAD_IX).cuda()
    opt             = torch.optim.Adam(comment_encoder.parameters(), lr=1e-3)

    for epoch_i in range(num_epochs):

        print("Training:")
        train_loss  = train_batches = 0
        train_acc   = 0
        total       = 0
        train_preds = []
        labels      = []

        
        comment_encoder.train(True)
        
        for batch in iterate_minibatches(data_train, token_to_id, max_batches=None, verbose=False, target_columns=target_columns):

            comments_ix = torch.LongTensor(batch['comment_text']).cuda()
            targets     = batch['target'].argmax(axis=1)
            reference   = torch.LongTensor(targets).cuda()

            prediction = comment_encoder(comments_ix)
            
            train_preds.append(prediction.cpu().detach().numpy())
            labels.append(batch['target'])
            
            loss     = compute_loss(prediction, reference)
            accuracy = compute_accuracy(reference, prediction)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            train_loss += loss.cpu().detach().data.numpy()
            train_acc  += accuracy
            
            train_batches += 1
            total         += reference.size(0)
        
        print("\tLoss:\t%.5f" % (train_loss / train_batches))
        print("\tAcc:\t%.5f" % (train_acc / total))
        print('\n\n')
        
        print("Validation:")
        val_loss = val_batches = 0
        val_acc  = 0
        total    = 0
        
        comment_encoder.train(False)
        
        for batch in iterate_minibatches(data_val, token_to_id, shuffle=False, verbose=False, target_columns=target_columns):
            comments_ix = torch.LongTensor(batch['comment_text']).cuda()
            targets     = batch['target'].argmax(axis=1)
            reference   = torch.LongTensor(targets).cuda()

            prediction = comment_encoder(comments_ix)
            loss       = compute_loss(prediction, reference)
            accuracy   = compute_accuracy(reference, prediction)
            
            val_loss   += loss.cpu().detach().data.numpy()
            val_acc    += accuracy
            
            val_batches += 1
            total       += reference.size(0)
            
        print("\tLoss:\t%.5f" % (val_loss / val_batches))
        print("\tAcc:\t%.5f" % (val_acc / total))
        print('\n\n')
        
        train_preds = np.vstack(train_preds)
        labels      = np.vstack(labels)
        
        col_auc = []
        
        # train_probs = F.softmax(torch.tensor(train_preds), dim=1)
        train_probs = train_preds

        for i in range(train_preds.shape[1]):
            col_auc.append(roc_auc_score(labels[:, i], train_probs[:, i]))
            
        print('Mean AUC: {}'.format(np.mean(col_auc)))


def train_and_submit(train, test, sub):
    tokens, token_to_id  = preprocess(train)
    target_columns       = train.columns.tolist()[2:]

    UNK, PAD       = 'UNK', 'PAD'
    UNK_IX, PAD_IX = map(token_to_id.get, [UNK, PAD])

    comment_encoder = CommentsEncoder(out_size=64, n_tokens=len(tokens), PAD_IX=PAD_IX).cuda()
    opt             = torch.optim.Adam(comment_encoder.parameters(), lr=1e-3)

    for epoch_i in range(num_epochs):

        print("Training:")
        
        comment_encoder.train(True)
        
        for batch in iterate_minibatches(train, token_to_id, max_batches=None, verbose=False, target_columns=target_columns):

            comments_ix = torch.LongTensor(batch['comment_text']).cuda()
            targets     = batch['target'].argmax(axis=1)
            reference   = torch.LongTensor(targets).cuda()

            prediction = comment_encoder(comments_ix)
            loss       = compute_loss(prediction, reference)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
        
    print('\n\n')
    
    print("Validation:")
    comment_encoder.train(False)
    val_preds = []

    for batch in iterate_minibatches(test, token_to_id, shuffle=False, verbose=False, target_columns=target_columns):
        comments_ix = torch.LongTensor(batch['comment_text']).cuda()
        prediction  = comment_encoder(comments_ix)
        val_preds.append(prediction.cpu().detach().numpy())
    
    val_preds       = np.vstack(val_preds)
    # sub.iloc[:, 1:] = F.softmax(torch.tensor(val_preds), dim=1)
    sub.iloc[:, 1:] = val_preds


    sub.to_csv(f'../submissions/{filename}.csv', index=False)

def main(train, test, sub, cv=True, filename='baseline_cnn'):
    if cv:
        train_with_cv(train)
    else:
        train_and_save(train, test, sub, filename=filename)


if __name__ == "__main__":
    train, test = load_data()
    sub         = load_sub()

    main(train, test, sub, filename=f'baseline_cnn_{num_epochs}')