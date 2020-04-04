import collections.abc

class TreebankNode(object):
    '''
    parent class for all kinds of TreeBank objects
    '''
    pass

class InternalTreebankNode(TreebankNode):
    '''
    represents a label for a part of the sentence (s[i:j], i != j)
    '''
    def __init__(self, label, children):
        self.label = label
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label) 
            # when a part of the tree in linear , and I think this happens only when the internal node
            # has "no label (phi)". that is why the tree also gets assigned to its only child, until it has more than one child

        children = []
        # this recursively assigns indices to children. The left most leaf gets (0,1), and then the next leaf (1,2) and so on.
        # the internal nodes get (left most index in subtree, right most index in subtree)
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right # the index of the rightmost child

        return InternalParseNode(tuple(sublabels), children)

class LeafTreebankNode(TreebankNode):
    '''
    represents one word in the sentence, along with it's tag
    '''
    def __init__(self, tag, word):
        self.tag = tag
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        self.label = label
        self.children = tuple(children)
        self.left = children[0].left  #index of the left most leaf in sub tree
        self.right = children[-1].right # index of right most leaf in sub tree

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        # returns internal node for s[left : right]
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        # returns label for s[left : right]
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        # returns the index at which the current part of sentence has been split
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        self.left = index
        self.right = index + 1
        self.tag = tag
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

def load_trees(path, strip_top=True):
    '''
    A recursive function used to load the tree from the dataset. 

    Base Case : create a LeafTreebankNode object when there are no more children (interpreted as 
    when no opening "(" bracket is encountered)

    Recursion : creates an InternalTreebankNode with the label for that past of the sentence and 
    calls recursively for the children 
    '''
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()
    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                trees[i] = tree.children[0]

    return trees
    

