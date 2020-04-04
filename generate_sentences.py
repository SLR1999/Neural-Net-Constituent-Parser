import trees

train_treebank = trees.load_trees("data/02-21.10way.clean")

with open("data/02-21_sentence.txt",'w',encoding = 'utf-8') as f:
    for sentence in range(len(train_treebank)):
        f.write(train_treebank[sentence].sentencify() + "\n")

f.close()