import trees

train_treebank = trees.load_trees("02-21.10way.clean")

print("Loaded {:,} training examples.".format(len(train_treebank)))

print(train_treebank[0])