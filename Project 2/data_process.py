from subprocess import PIPE, Popen
import os
import random
from lang import Lang
from helper import indexes_from_sentence

import torch
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

num_merge_operation = 10000

MIN_LENGTH = 3
MAX_LENGTH = 50

PAD_token = 0
SOS_token = 1
EOS_token = 2


def load_data(data):
    with open(data, 'r', encoding='utf8') as f:
        return f.read().splitlines()

def write_file(data, file):
    file = open(file, 'w')
    for sent in data:
        sent = " ".join(str(w) for w in sent)
        file.write(sent)
        file.write('\n')
    file.close()


def revert_BPE(sentence):
    sentence = sentence.replace("<EOS>", "")
    command = 'echo "{}" | sed -E "s/(@@ )|(@@ ?$)//g"'.format(sentence)
    give_command = Popen(args=command, stdout=PIPE, shell=True).communicate()
    reverted_sentence = give_command[0]
    return reverted_sentence.decode('utf-8', 'ignore')


# Pad with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def get_batches(input_lang, target_lang, batch_size, pairs, index):

    input_seqs = []
    target_seqs = []

    for j in range(index, index+batch_size):
        pair = pairs[j]
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(target_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


def random_batch(input_lang, target_lang, batch_size, pairs):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(target_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


def process_sentences(input_lang, target_lang, input_sentences, target_sentences, reduce_complexity=False):
    input_obj = Lang(input_lang)  # in our case french
    output_obj = Lang(target_lang)  # mostly english

    print("Reading sentence pairs...")
    # create parallel sentence pairs
    pairs = list(zip(input_sentences, target_sentences))
    if reduce_complexity:
        for sentence in pairs:
            if len(sentence[0]) >= MIN_LENGTH and len(sentence[0]) <= MAX_LENGTH \
                and len(sentence[1]) >= MIN_LENGTH and len(sentence[1]) <= MAX_LENGTH:
                input_sent = sentence[0].strip().split(' ')
                target_sent = sentence[1].strip().split(' ')
                input_obj.add_sentence(input_sent)
                output_obj.add_sentence(target_sent)
    else:
        for sentence in pairs:
            input_sent = sentence[0].strip().split(' ')
            target_sent = sentence[1].strip().split(' ')
            input_obj.add_sentence(input_sent)
            output_obj.add_sentence(target_sent)

    input_obj.add_word('UNK')
    output_obj.add_word('UNK')

    return input_obj, output_obj, pairs


def preprocess_data():
    # TODO: Try BPE heuristics since english and french share the alphabets
    # TODO: Try also convert rare words into character n-grams
    # XXX: For BPE first install the following,
    # pip install subword-nmt
    # pip install https://github.com/rsennrich/subword-nmt/archive/master.zip

    print("Starting Preprocessing....")

    # Pre-processing English corpora
    print("Processing English language corpora....")
    print("Processing training set....")
    print("Processing validation set....")
    print("Processing testing set....")
    # Tokenize
    os.system("perl tokenizer.perl -l en < ./data/val/val.en > ./data/val/val_tok.en")
    os.system("perl tokenizer.perl -l en < ./data/train/train.en > ./data/train/train_tok.en")
    os.system("perl tokenizer.perl -l en < ./data/test/test.en > ./data/test/test_tok.en")
    print("Done Tokenizing!!")
    # Lowercasing
    os.system("perl lowercase.perl -l en < ./data/val/val_tok.en > ./data/val/val_lower.en")
    os.system("perl lowercase.perl -l en < ./data/train/train_tok.en > ./data/train/train_lower.en")
    os.system("perl lowercase.perl -l en < ./data/test/test_tok.en > ./data/test/test_lower.en")
    print("Done Lowercasing!!")
    # BPE
    os.system(
        "subword-nmt learn-bpe -s {} < ./data/val/val_lower.en > ./data/val/val_code.en.bpe".format(
            num_merge_operation))
    os.system("subword-nmt apply-bpe -c ./data/val/val_code.en.bpe < ./data/val/val_lower.en > ./data/val/val_final.en")
    os.system(
        "subword-nmt learn-bpe -s {} < ./data/train/train_lower.en > ./data/train/train_code.en.bpe".format(
            num_merge_operation))
    os.system(
        "subword-nmt apply-bpe -c ./data/train/train_code.en.bpe < ./data/train/train_lower.en > ./data/train/train_final.en")
    os.system(
        "subword-nmt learn-bpe -s {} < ./data/test/test_lower.en > ./data/test/test_code.en.bpe".format(
            num_merge_operation))
    os.system(
        "subword-nmt apply-bpe -c ./data/test/test_code.en.bpe < ./data/test/test_lower.en > ./data/test/test_final.en")
    print("Done BPE!!")

    # Pre-processing French corpora
    print("Processing French language corpora....")
    print("Processing training set....")
    print("Processing validation set....")
    print("Processing testing set....")
    # Tokenize
    os.system("perl tokenizer.perl -l fr < ./data/val/val.fr > ./data/val/val_tok.fr")
    os.system("perl tokenizer.perl -l fr < ./data/train/train.fr > ./data/train/train_tok.fr")
    os.system("perl tokenizer.perl -l fr < ./data/test/test.fr > ./data/test/test_tok.fr")
    print("Done Tokenizing!!")
    # Lowercasing
    os.system("perl lowercase.perl -l fr < ./data/val/val_tok.fr > ./data/val/val_lower.fr")
    os.system("perl lowercase.perl -l fr < ./data/train/train_tok.fr > ./data/train/train_lower.fr")
    os.system("perl lowercase.perl -l fr < ./data/test/test_tok.fr > ./data/test/test_lower.fr")
    print("Done Lowercasing!!")
    # BPE
    os.system(
        "subword-nmt learn-bpe -s {} < ./data/val/val_lower.fr > ./data/val/val_code.fr.bpe".format(
            num_merge_operation))
    os.system("subword-nmt apply-bpe -c ./data/val/val_code.fr.bpe < ./data/val/val_lower.fr > ./data/val/val_final.fr")
    os.system(
        "subword-nmt learn-bpe -s {} < ./data/train/train_lower.fr > ./data/train/train_code.fr.bpe".format(
            num_merge_operation))
    os.system(
        "subword-nmt apply-bpe -c ./data/train/train_code.fr.bpe < ./data/train/train_lower.fr > ./data/train/train_final.fr")
    os.system(
        "subword-nmt learn-bpe -s {} < ./data/test/test_lower.fr > ./data/test/test_code.fr.bpe".format(
            num_merge_operation))
    os.system(
        "subword-nmt apply-bpe -c ./data/test/test_code.fr.bpe < ./data/test/test_lower.fr > ./data/test/test_final.fr")
    print("Done BPE!!")

    print("Done Preprocessing!!!")



