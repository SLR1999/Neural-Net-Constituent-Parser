import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Feedforward, self).__init__()
        
        self.weights = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(nn.Linear(prev_dim, next_dim))

        #the above should be nn.Parameters I guess

        #implement forward function where we need to perform affine transformations

class TopDownParser(nn.Module):
    def __init__(
            self,
            model,
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

        self.tag_embeddings = nn.Embedding(tag_vocab.size, tag_embedding_dim)
        self.word_embeddings = nn.Embedding(word_vocab.size, word_embedding_dim)

        #self.lstm ??

        self.f_label = Feedforward(2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(2 * lstm_dim, [split_hidden_dim], 1)

        #don't know, will figure out
