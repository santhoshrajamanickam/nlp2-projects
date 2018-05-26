import os
from positionalEncoder import PositionalEncoder
from attentionDecoder import AttnDecoderRNN, BahdanauAttnDecoderRNN, LuongAttnDecoderRNN
from models import Model
from helper import get_all_variables
from rnnEncoder import EncoderRNN
from data_process import load_data, process_sentences, random_batch, revert_BPE
from masked_cross_entropy import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_COUNT = 5
BATCH_SIZE = 2

PAD_token = 0
SOS_token = 1
EOS_token = 2

# hyperparameters
word_embedding_size = 500
pos_embedding_size = 50
hidden_size = word_embedding_size + pos_embedding_size
maximum_length = 50

train_en_sentences = load_data('./data/train/train_final.en')
train_fr_sentences = load_data('./data/train/train_final.fr')

french, english, pairs = process_sentences('french', 'english', train_fr_sentences,
                                           train_en_sentences, reduce_complexity=True)

# Something to try out
# french.trim(MIN_COUNT)
# english.trim(MIN_COUNT)

input_voc_size = french.n_words
output_voc_size = english.n_words

print("Size of English vocabulary: ", output_voc_size)
print("Size of French vocabulary: ", input_voc_size)

# print(random_batch(french, english, BATCH_SIZE, pairs))
# Testing on a small example
small_batch_size = 3
input_batches, input_lengths, target_batches, target_lengths = random_batch(french, english, small_batch_size, pairs)


small_hidden_size = 8
small_n_layers = 2

encoder_test = EncoderRNN(french.n_words, small_hidden_size, small_n_layers)
decoder_test = LuongAttnDecoderRNN(small_hidden_size, english.n_words, small_n_layers)

encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)
print('encoder_outputs', encoder_outputs.size()) # max_len x batch_size x hidden_size
print('encoder_hidden', encoder_hidden.size()) # n_layers * 2 x batch_size x hidden_size

max_target_length = max(target_lengths)

# Prepare decoder input and outputs
decoder_input = Variable(torch.LongTensor([SOS_token] * small_batch_size, device=device))
decoder_hidden = encoder_hidden[:decoder_test.n_layers] # Use last (forward) hidden state from encoder
all_decoder_outputs = Variable(torch.zeros(max_target_length, small_batch_size, decoder_test.output_size, device=device))

# Run through decoder one time step at a time
for t in range(max_target_length):
    decoder_output, decoder_hidden, decoder_attn = decoder_test(
        decoder_input, decoder_hidden, encoder_outputs
    )
    all_decoder_outputs[t] = decoder_output # Store this step's outputs
    decoder_input = target_batches[t] # Next input is current target

# Test masked cross entropy loss
loss = masked_cross_entropy(
    all_decoder_outputs.transpose(0, 1).contiguous(),
    target_batches.transpose(0, 1).contiguous(),
    target_lengths
)
print('loss', loss.item())



#
# positional_encoder = PositionalEncoder(input_voc_size, word_embedding_size, pos_embedding_size, maximum_length)
# decoder = AttnDecoderRNN(hidden_size, output_voc_size, maximum_length)
# # positional_encoder.load_state_dict('./models/positional_encoder.pt')
# # decoder.load_state_dict('./models/positional_decoder.pt')
#
# # n_iters = 2000
# positional_model = Model('positional', french, english, tensor_pairs, positional_encoder, decoder, maximum_length)
# for i in range(0, 10):
#     loss, time_taken = positional_model.epoch(limit=10)
#     print('Epoch {}: Loss:{} Time_taken:{}'.format(i, loss, time_taken))
#     positional_model.translate('./data/val/val_final.fr', './val_predictions.txt')
#
# # positional_model.save_model()
# positional_model.translate('./data/test/test_final.fr', './test_predictions.txt')
#
# # os.system("perl multi-bleu.perl -lc ./data/test/test_final.en < test_predictions.txt")
#
# # positional_model = Model('positional', french, english, tensor_pairs, positional_encoder, decoder, maximum_length)