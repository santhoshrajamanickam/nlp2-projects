import os
from positionalEncoder import PositionalEncoder
from attentionDecoder import AttnDecoderRNN
from models import Model
from data_process import load_data, process_sentences, revert_BPE #, tokenize


# hyperparameters
word_embedding_size = 500
pos_embedding_size = 50
hidden_size = word_embedding_size + pos_embedding_size
maximum_length = 50

train_en_sentences = load_data('./data/train/train_final.en')
train_fr_sentences = load_data('./data/train/train_final.fr')

# print("Nr of English sentences: ", len(train_en_sentences))
# print("Nr of French sentences: ", len(train_fr_sentences))

french, english, pairs = process_sentences('french', 'english', train_fr_sentences, train_en_sentences)

# # example print of revert BPE
# test = train_en_sentences[19]
# print(test)
# print(revert_BPE(test))

input_voc_size = french.n_words
output_voc_size = english.n_words

print("Size of English vocabulary: ", output_voc_size)
print("Size of French vocabulary: ", input_voc_size)


positional_encoder = PositionalEncoder(input_voc_size, word_embedding_size, pos_embedding_size, maximum_length)
decoder = AttnDecoderRNN(hidden_size, output_voc_size, maximum_length)

n_iters = 2000
positional_model = Model('positional', french, english, pairs, positional_encoder, decoder, maximum_length)
positional_model.epoch(n_iters)
positional_model.save_model()
positional_model.translate('./data/test/test_final.fr')

os.system("perl multi-bleu.perl -lc ./data/test/test_final.en < test_predictions.txt")