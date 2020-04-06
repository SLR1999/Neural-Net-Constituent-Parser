import functools

import dynet as dy
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    shape = scores.size()[0]
    increment = np.ones(shape)
    increment[oracle_index] = 0
    increment = torch.FloatTensor(increment)
    return scores + increment

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_dim, hidden_dim)
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.output(F.relu(self.hidden(x)))
        return x 


class TopDownParser(nn.Module):

    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
    ):
        super(TopDownParser, self).__init__()
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = nn.Parameter(torch.zeros(tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = nn.Parameter(torch.zeros(word_vocab.size, word_embedding_dim))

        self.lstm = nn.LSTM(
            input_size = tag_embedding_dim + word_embedding_dim, 
            hidden_size = lstm_dim,
            num_layers = lstm_layers,
            dropout = dropout,
            bidirectional=True
        )

        # two NN for two independent parts (as per the paper) - getting the label 
        # for the whole span and the split for the span
        self.f_label = Network(2 * lstm_dim, label_hidden_dim, label_vocab.size)
        self.f_split = Network(2 * lstm_dim, split_hidden_dim, 1)

        self.dropout = dropout

    def parse(self, sentence, gold=None, explore=True):

        is_train = gold is not None

        if is_train:
            #enable dropout in lstm
            pass
        else:
            #disable dropout in lstm
            torch.no_grad()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embedding = [torch.cat((tag_embedding,word_embedding),0).tolist()]
            embeddings.append(embedding)

        embeddings = torch.tensor(embeddings)
        lstm_outputs,_ = self.lstm(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][0][:self.lstm_dim] -
                lstm_outputs[left][0][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][0][self.lstm_dim:] -
                lstm_outputs[right + 1][0][self.lstm_dim:])
            return torch.tensor(torch.cat((forward,backward),0).tolist())

        def helper(left, right):
            # print("left : ", left , "right", right)
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))
            label_scores.requires_grad_(True)

            if is_train:
                oracle_label = gold.oracle_label(left, right) 
                # gets the correct label for s[left : right]
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index) 

            label_scores_np = label_scores.detach().numpy()
            # npvalue : Returns the value of the expression as a numpy array
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1) # this part also me confused
            argmax_label = self.label_vocab.value(argmax_label_index)

            # numpy.argmax(array, axis = None, out = None) : Returns indices of the max 
            # element of the array in a particular axis.

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else torch.zeros(1))
            else:
                label = argmax_label
                label_loss = label_scores[argmax_label_index]

            if right - left == 1:
                tag, word = sentence[left]
                tree = trees.LeafParseNode(left, tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right):
                left_encodings.append(get_span_encoding(left, split).tolist())
                right_encodings.append(get_span_encoding(split, right).tolist())
            left_scores = torch.tensor([self.f_split(torch.tensor(encoding)).item() for encoding in left_encodings])
            right_scores = torch.tensor([self.f_split(torch.tensor(encoding)).item() for encoding in right_encodings])
            split_scores = left_scores + right_scores
            split_scores.requires_grad_(True)
            # print("split scores : ")
            # print(split_scores)
            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            split_scores_np = split_scores.detach().numpy()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)
            # print(argmax_split)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else torch.zeros(1))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            # print("left = ", left , "split = ", split, "right = ", right)
            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            label_loss.requires_grad_(True)
            split_loss.requires_grad_(True)
            left_loss.requires_grad_(True)
            right_loss.requires_grad_(True)
            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        return tree, loss
