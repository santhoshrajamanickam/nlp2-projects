import time
import random
import math

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from helper import variables_from_pair, variable_from_sentence, timeSince
from data_process import load_data, revert_BPE

SOS_token = 0
EOS_token = 1


class Model:

    def __init__(self, model_name, L1, L2, sentence_pairs, encoder, decoder, max_length, learning_rate=0.01, use_pretrained=False):
        self.model_name = model_name
        self.L1 = L1
        self.L2 = L2
        self.pairs = sentence_pairs
        if not use_pretrained:
            self.encoder = encoder
            self.decoder = decoder
        else:
            self.encoder = torch.load('./models/{}_encoder.pt'.format(self.model_name))
            self.decoder = torch.load('./models/{}_decoder.pt'.format(self.model_name))
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.max_length = max_length
        self.criterion = nn.NLLLoss()

    def save_model(self):
        # save models
        torch.save(self.encoder.state_dict(), './models/{}_encoder.pt'.format(self.model_name))
        torch.save(self.decoder.state_dict(), './models/{}_decoder.pt'.format(self.model_name))

    def train(self, input_variable, target_variable, teacher_forcing_ratio = 0.5, clip = 5.0):

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word

        # Get size of input and target sentences
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([[SOS_token]], device=device))
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        if use_teacher_forcing:

            # Teacher forcing: Use the ground-truth target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Next target is next input

        else:
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])

                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]], device=device))  # Chosen word is next input

                # Stop at end of sentence (not necessary when using known targets)
                if ni == EOS_token: break

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def epoch(self, n_iters, print_every=100, plot_every=100):
        print('===============Training model...====================')
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        # plot_loss_total = 0  # Reset every plot_every

        for i in range(0, n_iters):

            input_variable, target_variable = variables_from_pair(self.L1, self.L2, random.choice(self.pairs))
            
            if len(input_variable) > self.max_length:
                continue
            loss = self.train(input_variable, target_variable)
            print_loss_total += loss

            if i % print_every == 0 and i != 0:
                print(i)
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                             i, i / n_iters * 100, print_loss_avg))


    def predict(self, sentence):
        with torch.no_grad():
            input_variable = variable_from_sentence(self.L1, sentence)
            input_length = input_variable.size()[0]
            encoder_output, encoder_hidden = self.encoder(input_variable)

            decoder_input = Variable(torch.LongTensor([[SOS_token]], device=device))
            decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_output)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.L2.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def translate(self, input_path):
        sentences = load_data(input_path)
        with open('test_predictions.txt', 'w') as file:
            for sentence in sentences:
                prediction, _ = self.predict(sentence)
                sentence = (' '.join(prediction).replace('"', ''))
                translation = revert_BPE(sentence)
                file.write(str(translation))
