from model import *
from train import *
from lang import Lang
from data_process import load_data, tokenize

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
word_embedding_size = 256
pos_embedding_size = 20
hidden_size = word_embedding_size + pos_embedding_size
maximum_length = 50

english_data = load_data('data/train/train_complete.en')
french_data = load_data('data/train/train_complete.fr')


# use Lang datastructure from Pytorch seq2seq tutorial
english = Lang(english_data)
french = Lang(french_data)

# create parallel sentence pairs
sentences = list(zip(english_data, french_data))
for sent in sentences:
    english.addSentence(sent[0])
    french.addSentence(sent[1])

input_voc_size = english.n_words
output_voc_size = french.n_words

encoder = PositionalEncoder(input_voc_size, word_embedding_size, pos_embedding_size, maximum_length)
decoder = AttnDecoderRNN(hidden_size, output_voc_size, maximum_length)

print('===============Training model...====================')
n_iters = 100
epoch(english, french, sentences, encoder, decoder, n_iters, maximum_length, print_every=50)
