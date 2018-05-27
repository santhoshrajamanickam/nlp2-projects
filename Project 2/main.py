import os
import torch
from rnnEncoder import EncoderRNN
from attentionDecoder import LuongAttnDecoderRNN
from models import Model
from data_process import load_data, process_sentences, get_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

use_pretrained = True

# hyperparameters
hidden_size = 500
n_layers = 2
batch_size = 50
learning_rate = 0.01
decoder_learning_ratio = 5.0
no_epochs = 5


train_en_sentences = load_data('./data/train/train_final.en')
train_fr_sentences = load_data('./data/train/train_final.fr')

french, english, pairs = process_sentences('french', 'english', train_fr_sentences,
                                           train_en_sentences, reduce_complexity=True)

input_voc_size = french.n_words
output_voc_size = english.n_words

print("Size of English vocabulary: ", output_voc_size)
print("Size of French vocabulary: ", input_voc_size)

encoder = EncoderRNN(french.n_words, hidden_size, n_layers)
decoder = LuongAttnDecoderRNN(hidden_size, english.n_words, n_layers)

if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

if os.path.isfile('./models/rnn_model_encoder.pt') and use_pretrained:
    encoder.load_state_dict(torch.load('./models/rnn_model_encoder.pt'))
if os.path.isfile('./models/rnn_model_decoder.pt') and use_pretrained:
    decoder.load_state_dict(torch.load('./models/rnn_model_decoder.pt'))

rnn_model = Model('rnn', french, english, pairs, encoder, decoder,
                  learning_rate=learning_rate, use_batching=True, batch_size=batch_size)

rnn_model.random_epoch(100)
rnn_model.save_model('./models/rnn_model_encoder.pt', './models/rnn_model_decoder.pt')
print('===============Testing model...====================')
rnn_model.translate('./data/test/test_final.fr', './testing/test_predictions.txt')
os.system("perl multi-bleu.perl -lc ./data/test/test_final.en < ./testing/test_predictions.txt")
