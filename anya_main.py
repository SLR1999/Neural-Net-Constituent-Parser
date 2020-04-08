import argparse
import itertools
import os.path
import time

import dynet as dy
import numpy as np
import torch

import evaluate
import anya_parse as parse
import trees
import vocabulary

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    parser = parse.TopDownParser(
        tag_vocab,
        word_vocab,
        label_vocab,
        args.tag_embedding_dim,
        args.word_embedding_dim,
        args.lstm_layers,
        args.lstm_dim,
        args.label_hidden_dim,
        args.split_hidden_dim,
        args.dropout,
    )
    # trainer = dy.AdamTrainer(model)
    learning_rate = 1e-4
    trainer = torch.optim.Adam(parser.parameters(), lr=learning_rate)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch

    start_time = time.time()

    def check_dev():
        best_dev_fscore = 0
        best_dev_model_path = None

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            # dy.renew_cg()
            trainer.zero_grad()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            predicted, _ = parser.parse(sentence)
            dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            torch.save(parser.state_dict(), best_dev_model_path)

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            trainer.zero_grad()
            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                _, loss = parser.parse(sentence, tree)
                loss.requires_grad_(True)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_losses = torch.tensor(batch_losses,requires_grad = True)
            batch_loss_value = torch.mean(batch_losses)
            batch_losses.mean().backward()
            trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    # model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    start_time = time.time()

    test_predicted = []
    for tree in test_treebank:
        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        predicted, _ = parser.parse(sentence)
        test_predicted.append(predicted.convert())

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-seed", type=int, default=50)
    parser.add_argument("--tag-embedding-dim", type=int, default=50)
    parser.add_argument("--word-embedding-dim", type=int, default=100)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dim", type=int, default=250)
    parser.add_argument("--label-hidden-dim", type=int, default=250)
    parser.add_argument("--split-hidden-dim", type=int, default=250)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--explore", action="store_true", default=True)
    parser.add_argument("--model-path-base", default="model/best.pt")
    parser.add_argument("--evalb-dir", default="EVALB/")
    parser.add_argument("--train-path", default="data/02-21.10way.clean")
    parser.add_argument("--dev-path", default="data/22.auto.clean")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checks-per-epoch", type=int, default=4)
    parser.add_argument("--print-vocabs", action="store_true")

    # parser = parsers.add_parser("test")
    # parser.set_defaults(callback=run_test)
    # parser.add_argument("--model-path-base", required=True)
    # parser.add_argument("--evalb-dir", default="EVALB/")
    # parser.add_argument("--test-path", default="data/23.auto.clean")

    args = parser.parse_args()
    # args.callback(args)
    run_train(args)

if __name__ == "__main__":
    main()
