from mosestokenizer import *
from subprocess import PIPE, Popen

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

def tokenize(data, language):
    tokenizer = MosesTokenizer(language)
    tokenized = [tokenizer(sent) for sent in data]
    tokenizer.close()
    return tokenized

def revert_BPE(sentence):
    sentence = sentence.replace("<EOS>", "")
    command = 'echo "{}" | sed -E "s/(@@ )|(@@ ?$)//g"'.format(sentence)
    give_command =  Popen(args=command,stdout=PIPE,shell=True).communicate()
    return give_command[0]

english_train = load_data('data/test/test_2017_flickr.en')
french_train = load_data('data/test/test_2017_flickr.fr')


english_tokenized = tokenize(english_train, 'en')
french_tokenized = tokenize(french_train, 'fr')


english_data = [[word.lower() for word in sent] for sent in english_tokenized]
french_data = [[word.lower() for word in sent] for sent in french_tokenized]

write_file(english_data, 'data/test/test_clean.en')
write_file(french_data, 'data/test/test_clean.fr')
