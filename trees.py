import collections.abc

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
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

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

class LeafTreebankNode(TreebankNode):
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

        self.left = children[0].left
        self.right = children[-1].right

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
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
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