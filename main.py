import trees

train_treebank = trees.load_trees("data/02-21.10way.clean")

print("Loaded {:,} training examples.".format(len(train_treebank)))

s = train_treebank[11]
print(s)
t = s.sentencify()
print(t)