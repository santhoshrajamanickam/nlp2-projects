from subprocess import PIPE, Popen
import os

num_merge_operation = 10000

def load_data(data):
    with open(data, 'r') as f:
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


# TODO: Try BPE heuristics since english and french share the alphabets
# TODO: Try also convert rare words into character n-grams
# XXX: For BPE first install the following,
# pip install subword-nmt
# pip install https://github.com/rsennrich/subword-nmt/archive/master.zip


def preprocess():
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
        "subword-nmt learn-bpe -s {} < ./data/val/val_lower.en > ./data/val/val_code.en.bpe".format(num_merge_operation))
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
        "subword-nmt learn-bpe -s {} < ./data/val/val_lower.fr > ./data/val/val_code.fr.bpe".format(num_merge_operation))
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
