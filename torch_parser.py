# SSM's code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def augment(scores, oracle_index):
    print("Size of scores  tensor: {}".format(scores.size()))
    shape = list(scores.size())[0]
    increment = torch.ones(shape)
    # increment = torch.ones(shape,1)
    increment[oracle_index] = 0
    return scores + increment

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