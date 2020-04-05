# SSM's code
import torch
import torch.nn as nn
from torch.autograd import Variable

import functools
import numpy as np

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    """Adds the hamming loss to the wrong labels"""
    print("(Augment function) Size of scores tensor: {}".format(scores.size()))
    shape = list(scores.size())[0]
    increment = torch.ones(shape)
    # increment = torch.ones(shape,1)
    increment[oracle_index] = 0
    return scores + Variable(increment)

class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Feedforward, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        
        self.weights = []
        self.biases = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(nn.Parameter(torch.zeros(prev_dim, next_dim)))
            self.biases.append(nn.Parameter(torch.zeros(next_dim,1)))

    def forward(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            print(x)
            x = torch.matmul(weight.t(),x)
            print(x)
            x = torch.add(bias,x)
            if i < len(self.weights) - 1:
                x = self.relu(x)
            print(x)
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
        self.tag_embedding_dim = tag_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.lstm_layers = lstm_layers
        self.lstm_dim = lstm_dim
        self.label_hidden_dim = label_hidden_dim
        self.split_hidden_dim = split_hidden_dim
        self.dropout = dropout

        self.tag_embeddings = nn.Embedding(tag_vocab.size, tag_embedding_dim)
        self.word_embeddings = nn.Embedding(word_vocab.size, word_embedding_dim)

        self.lstm = nn.LSTM(
            self.tag_embedding_dim + self.word_embedding_dim,
            2 * self.lstm_dim,
            num_layers=self.lstm_layers,
            dropout=self.dropout,
            bidirectional=True
            )

        self.f_label = Feedforward(2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(2 * lstm_dim, [split_hidden_dim], 1)

    def forward(self, sentence, gold=None, explore=True):
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
            embeddings.append(torch.cat([tag_embedding, word_embedding],1))

        lstm_outputs = self.lstm(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return torch.cat([forward, backward],1)

        def helper(left, right):
            label_scores = self.f_label(get_span_encoding(left, right))

            if is_train:
                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index)

            label_scores_np = label_scores.numpy()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1)
            argmax_label = self.label_vocab.value(argmax_label_index)

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else Variable(torch.zeros(1)))
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
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))

            #need to check dimensions here
            print("(Helper function) Dimension of split encodings left: {}".format(left_encodings[0].size()))
            left_scores = self.f_split(left_encodings,1)
            right_scores = self.f_split(right_encodings,1)
            split_scores = left_scores + right_scores
            split_scores = split_scores.view(-1, len(left_encodings), 1)
            print("(Helper function) Dimension of split scores: {}".format(split_scores.size()))

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            split_scores_np = split_scores.numpy()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else Variable(torch.zeros(1)))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        return tree, loss