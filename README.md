# Neural-Net-Constituent-Parser

The assignment is basically to implement a neural network constituency parser for English using the Penn Tree Bank as the ground truth dataset. The following options are considered :

1. Use a scoring mechanism that computes the score s(i,j,c) for the segment of the sentence between words i and j forming a constituent of type (non-terminal) c.

2. Recursive neural network building a parse tree recursively. 

3.Using a transformer like-architecture along with recurrence (as in the Universal Transformer paper) where each 'iteration' through the transformer corresponds to one level of the parse tree. In each iteration we decode one layer of the parse tree and that forms the input for the next iteration loop.

There are other methods based on seq2seq models, pointer networks etc. The last one incidentally does not assume that the constituents are contiguous groups of words.
