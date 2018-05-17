from mosestokenizer import *


def process_data(data):
    with open(data, 'r') as f:
        return f.read().splitlines()

def write_file(data, file):
    file = open(file, 'w')
    for sent in data:
        sent = " ".join(str(w) for w in sent)
        file.write(sent)
        file.write('\n')
    file.close()

english_train = process_data('data/train/train.en')
french_train = process_data('data/train/train.fr')

english_tokenizer = MosesTokenizer('en')
french_tokenizer = MosesTokenizer('fr')

english_tokenized = [english_tokenizer(sent) for sent in english_train]
french_tokenized = [french_tokenizer(sent) for sent in french_train]

english_tokenizer.close()
french_tokenizer.close()

english_data = [[word.lower() for word in sent] for sent in english_tokenized]
french_data = [[word.lower() for word in sent] for sent in french_tokenized]

write_file(english_data, 'data/train/train_clean.en')
write_file(french_data, 'data/train/train_clean.fr')
