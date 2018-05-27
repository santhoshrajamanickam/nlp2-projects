import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import random
from random import shuffle
import pickle

from helper import *

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class PositionalEncoder(nn.Module):
    def __init__(self, vocabulary_size, word_embedding_size, pos_embedding_size, max_length, dropout_p = 0.1):
        super(PositionalEncoder,self).__init__()
        self.hidden_size = word_embedding_size
        self.max_length = max_length
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.word_embedding = nn.Embedding(vocabulary_size, word_embedding_size)
        self.pos_embedding = nn.Embedding(max_length, word_embedding_size)

    def forward(self, input, hidden):
        i = input[0]
        word = input[1]

        #word_embedded = self.word_embedding(word).view(1, 1, -1)
        #pos_embedded = self.pos_embedding(torch.tensor(i).view(1, 1, -1))

        word_embedded = self.word_embedding(word).view(1, 1, -1)
        pos_embedded = self.pos_embedding(torch.tensor([i], dtype=torch.long)).view(1, 1, -1)

        output = torch.cat((word_embedded.squeeze(0), pos_embedded.squeeze(0)), 1)
        output = self.linear(self.dropout(output))
        return output, hidden


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



def epochPos(fr, en, sentences, encoder, decoder, n_iters, max_length):
    losses = []
    start = time.time()

    for i in range(1, n_iters + 1):

        print(i-1)
        losses.append(train_sentences(fr,en,sentences,encoder,decoder,max_length))
        print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                     i, i / n_iters * 100, losses[i-1]))

        torch.save(encoder.state_dict(), 'models/POSencoder_{}'.format(i))
        torch.save(decoder.state_dict(), 'models/POSdecoder_{}'.format(i))

    with open('POSloss', 'wb') as fp:
        pickle.dump(losses, fp)


def train_sentences(fr,en,sentences, encoder, decoder, max_length, num_pairs = 10000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    indices = list(range(len(sentences)))
    random.shuffle(indices)

    criterion = nn.NLLLoss()

    for iter in range(1, num_pairs + 1):
        training_pair = tensorsFromPair(fr,en,sentences[indices[iter-1]])

        input_tensor = training_pair[0]
        if len(input_tensor) > max_length:
            continue
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
        print_loss_total += loss
        plot_loss_total += loss

    return loss


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, teacher_forcing_ratio = 0.5):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    target_length = target_tensor.size(0)
    encoder_hidden = encoder.initHidden()
    average_hidden = torch.zeros(1, encoder.hidden_size)

    input_length = input_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder((ei, input_tensor[ei]), encoder_hidden)
        average_hidden += encoder_output
        encoder_outputs[ei] = encoder_output[0, 0]
    encoder_hidden = (average_hidden/input_length).unsqueeze(0)

    decoder_input = torch.tensor([[0]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == 1:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
