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
    assert len(shape) == 1
    increment = np.ones(shape)
    increment[oracle_index] = 0
    return scores + torch.tensor(increment)

# class Feedforward(object):
#     def __init__(self, model, input_dim, hidden_dims, output_dim):
#         self.spec = locals() # The locals() function returns a dictionary containing the 
#                             #  variables defined in the local namespace. 
#         self.spec.pop("self")
#         self.spec.pop("model")

#         self.model = model.add_subcollection("Feedforward") 
#         # model or parameter collection : A ParameterCollection holds Parameters. 
#         # Use it to create, load and save parameters.

#         self.weights = []
#         self.biases = []
#         dims = [input_dim] + hidden_dims + [output_dim]
#         for prev_dim, next_dim in zip(dims, dims[1:]): # gets a tuple of (prev_dim, next_dim) since the second part is delayed by 1
#             self.weights.append(self.model.add_parameters((next_dim, prev_dim))) # creates weights initialized to 0 of dimension specified
#             self.biases.append(self.model.add_parameters(next_dim))

#     def param_collection(self):
#         return self.model

#     @classmethod
#     def from_spec(cls, spec, model):
#         return cls(model, **spec)

#     def __call__(self, x):
#         for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
#             weight = dy.parameter(weight)
#             bias = dy.parameter(bias)
#             x = dy.affine_transform([bias, weight, x])
#             if i < len(self.weights) - 1:
#                 x = dy.rectify(x) # RelU function 
#         return x



# class TopDownParser(object):
#     def __init__(
#             self,
#             model,
#             tag_vocab,
#             word_vocab,
#             label_vocab,
#             tag_embedding_dim,
#             word_embedding_dim,
#             lstm_layers,
#             lstm_dim,
#             label_hidden_dim,
#             split_hidden_dim, # number of parts the sentence is split into 
#             dropout,
#     ):
#         self.spec = locals()
#         self.spec.pop("self")
#         self.spec.pop("model")

#         self.model = model.add_subcollection("Parser")
#         self.tag_vocab = tag_vocab
#         self.word_vocab = word_vocab
#         self.label_vocab = label_vocab
#         self.lstm_dim = lstm_dim

#         self.tag_embeddings = self.model.add_lookup_parameters(
#             (tag_vocab.size, tag_embedding_dim))
#         self.word_embeddings = self.model.add_lookup_parameters(
#             (word_vocab.size, word_embedding_dim))

#         # Add a lookup parameter to the ParameterCollection with a given initializer (here 0)

#         # classdynet.BiRNNBuilder(num_layers, input_dim, hidden_dim, model, 
#         # rnn_builder_factory, builder_layers=None)
#         # Builder for BiRNNs that delegates two regular RNNs and wires them together.
#         self.lstm = dy.BiRNNBuilder(
#             lstm_layers,
#             tag_embedding_dim + word_embedding_dim,
#             2 * lstm_dim,
#             self.model,
#             dy.VanillaLSTMBuilder)

#         # two NN for two independent parts (as per the paper) - getting the label 
#         # for the whole span and the split for the span
#         self.f_label = Feedforward(
#             self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
#         self.f_split = Feedforward(
#             self.model, 2 * lstm_dim, [split_hidden_dim], 1)

#         self.dropout = dropout

#     def param_collection(self):
#         return self.model

#     @classmethod
#     def from_spec(cls, spec, model):
#         return cls(model, **spec)

#     def parse(self, sentence, gold=None, explore=True):
#         is_train = gold is not None

#         if is_train:
#             self.lstm.set_dropout(self.dropout)
#         else:
#             self.lstm.disable_dropout()

#         embeddings = []
#         for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
#             tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
#             if word not in (START, STOP):
#                 count = self.word_vocab.count(word)
#                 if not count or (is_train and np.random.rand() < 1 / (1 + count)):
#                     word = UNK
#             word_embedding = self.word_embeddings[self.word_vocab.index(word)]
#             embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

#         lstm_outputs = self.lstm.transduce(embeddings)
#         # transduce : returns the list of output Expressions obtained by adding the 
#         # given inputs to the current state, one by one, to both the forward and backward RNNs, 
#         # and concatenating.

#         @functools.lru_cache(maxsize=None)
#         def get_span_encoding(left, right):
#             forward = (
#                 lstm_outputs[right][:self.lstm_dim] -
#                 lstm_outputs[left][:self.lstm_dim])
#             backward = (
#                 lstm_outputs[left + 1][self.lstm_dim:] -
#                 lstm_outputs[right + 1][self.lstm_dim:])
#             return dy.concatenate([forward, backward])

#         def helper(left, right):
#             assert 0 <= left < right <= len(sentence)

#             label_scores = self.f_label(get_span_encoding(left, right))

#             if is_train:
#                 oracle_label = gold.oracle_label(left, right) 
#                 # gets the correct label for s[left : right]
#                 oracle_label_index = self.label_vocab.index(oracle_label)
#                 label_scores = augment(label_scores, oracle_label_index) # me confused here help pls :p

#             label_scores_np = label_scores.npvalue()
#             # npvalue : Returns the value of the expression as a numpy array
#             argmax_label_index = int(
#                 label_scores_np.argmax() if right - left < len(sentence) else
#                 label_scores_np[1:].argmax() + 1) # this part also me confused
#             argmax_label = self.label_vocab.value(argmax_label_index)

#             # numpy.argmax(array, axis = None, out = None) : Returns indices of the max 
#             # element of the array in a particular axis.

#             if is_train:
#                 label = argmax_label if explore else oracle_label
#                 label_loss = (
#                     label_scores[argmax_label_index] -
#                     label_scores[oracle_label_index]
#                     if argmax_label != oracle_label else dy.zeros(1))
#             else:
#                 label = argmax_label
#                 label_loss = label_scores[argmax_label_index]

#             if right - left == 1:
#                 tag, word = sentence[left]
#                 tree = trees.LeafParseNode(left, tag, word)
#                 if label:
#                     tree = trees.InternalParseNode(label, [tree])
#                 return [tree], label_loss

#             left_encodings = []
#             right_encodings = []
#             for split in range(left + 1, right):
#                 left_encodings.append(get_span_encoding(left, split))
#                 right_encodings.append(get_span_encoding(split, right))
#             left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
#             right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
#             split_scores = left_scores + right_scores
#             split_scores = dy.reshape(split_scores, (len(left_encodings),))

#             if is_train:
#                 oracle_splits = gold.oracle_splits(left, right)
#                 oracle_split = min(oracle_splits)
#                 oracle_split_index = oracle_split - (left + 1)
#                 split_scores = augment(split_scores, oracle_split_index)

#             split_scores_np = split_scores.npvalue()
#             argmax_split_index = int(split_scores_np.argmax())
#             argmax_split = argmax_split_index + (left + 1)

#             if is_train:
#                 split = argmax_split if explore else oracle_split
#                 split_loss = (
#                     split_scores[argmax_split_index] -
#                     split_scores[oracle_split_index]
#                     if argmax_split != oracle_split else dy.zeros(1))
#             else:
#                 split = argmax_split
#                 split_loss = split_scores[argmax_split_index]

#             left_trees, left_loss = helper(left, split)
#             right_trees, right_loss = helper(split, right)

#             children = left_trees + right_trees
#             if label:
#                 children = [trees.InternalParseNode(label, children)]

#             return children, label_loss + split_loss + left_loss + right_loss

#         children, loss = helper(0, len(sentence))
#         assert len(children) == 1
#         tree = children[0]
#         if is_train and not explore:
#             assert gold.convert().linearize() == tree.convert().linearize()
#         return tree, loss

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

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = nn.Parameter(torch.zeros(tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = nn.Parameter(torch.zeros(word_vocab.size, word_embedding_dim))
        self.lstm = nn.LSTM(

            input_size = tag_embedding_dim + word_embedding_dim, 
            hidden_size = 2 * lstm_dim,
            num_layers = lstm_layers,
            bidirectional=True
        )

        # two NN for two independent parts (as per the paper) - getting the label 
        # for the whole span and the split for the span
        self.f_label = Network(2 * lstm_dim, label_hidden_dim, label_vocab.size)
        self.f_split = Network(2 * lstm_dim, split_hidden_dim, 1)

        self.dropout = dropout

    def parse(self, sentence, gold=None, explore=True):

        is_train = gold is not None

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embedding = [tag_embedding + word_embedding]
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
            return torch.tensor([torch.cat((forward,backward),0).tolist()])

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))[0]

            if is_train:
                oracle_label = gold.oracle_label(left, right) 
                # gets the correct label for s[left : right]
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index) 

            label_scores_np = np.array(label_scores)
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
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))
            left_scores = self.f_split(left_encodings)
            right_scores = self.f_split(right_encodings)
            split_scores = left_scores + right_scores
            split_scores = split_scores[0]

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            split_scores_np = np.array(split_scores)
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else torch.zeros(1))
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
