import time

import torch
from torch import optim
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    USE_CUDA = True

from helper import variables_from_pair, variable_from_sentence, time_since, indexes_from_sentence, as_minutes
from data_process import load_data, revert_BPE, get_batches
from masked_cross_entropy import *

SOS_token = 0
EOS_token = 1


class Model:

    def __init__(self, model_name, L1, L2, sentence_pairs, encoder, decoder, max_length=50, learning_rate=0.01, use_pretrained=False, use_batching=False, batch_size=1,decoder_learning_ratio=5.0):
        self.model_name = model_name
        self.L1 = L1
        self.L2 = L2
        self.batch_size = batch_size
        self.pairs = sentence_pairs
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        self.criterion = masked_cross_entropy

    def train(self, input_batches, input_lengths, target_batches, target_lengths, clip=5.0):

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word

        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # Use last (forward) hidden state from encoder

        max_target_length = max(target_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, self.batch_size, self.decoder.output_size))

        if USE_CUDA:
            all_decoder_outputs.cuda()
            decoder_input.cuda()
            encoder_outputs.cuda()


        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next input is current target

        # Loss calculation and backpropagation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item(), ec, dc

    def epoch(self, n_epochs, print_every=25):
        print('===============Training model...====================')
        epoch = 0

        while epoch < n_epochs:

            start = time.time()

            print_loss_total = 0
            num_batches = len(self.pairs) / self.batch_size

            batch = 1

            for index in range(0, len(self.pairs), self.batch_size):
                input_batch, input_lengths, target_batch, target_lengths = get_batches(self.L1, self.L2,
                                                                                       self.batch_size, self.pairs, index)

                # Run the train function
                loss, ec, dc = self.train(input_batch, input_lengths, target_batch, target_lengths)

                # Keep track of loss
                print_loss_total += loss
                print(batch)
                batch += 1

                if batch % print_every == 0 and batch != 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print_summary = '%s (%d %d%%) %.4f' % (
                    time_since(start, batch / num_batches), batch, batch / num_batches * 100, print_loss_avg)
                    print(print_summary)

            epoch += 1
            print('Done Epoch{} in {}'.format(epoch, as_minutes(start-time.time())))
            print('===============Validating model...====================')
            self.translate('./data/val/val_final.fr', './validation/val_predictions_{}.txt'.format(epoch))

    def evaluate(self, input_seq, max_length):
        with torch.no_grad():

            input_lengths = [len(input_seq)]
            input_seqs = [indexes_from_sentence(self.L1, input_seq)]
            input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)

            if USE_CUDA:
                input_batches.cuda()

            # Set to not-training mode to disable dropout
            self.encoder.train(False)
            self.decoder.train(False)

            # Run through encoder
            encoder_outputs, encoder_hidden = self.encoder.single_forward(input_batches, input_lengths, None)

            # Create starting vectors for decoder
            decoder_input = Variable(torch.LongTensor([SOS_token]))  # SOS
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # Use last (forward) hidden state from encoder

            if USE_CUDA:
                decoder_input.cuda()
                decoder_hidden.cuda()
                encoder_outputs.cuda()

            # Store output words and attention states
            decoded_words = []
            decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

            # Run through decoder
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                if ni == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.L2.index2word[ni.item()])

                # Next input is chosen word
                decoder_input = Variable(torch.LongTensor([ni]))
                if USE_CUDA: decoder_input.cuda()

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

    def translate(self, input_path, output_path):
        sentences = load_data(input_path)
        with open(output_path, 'w', encoding='utf8') as file:
            for sentence in sentences:
                prediction, _ = self.evaluate(sentence, max_length=100)
                sentence = (' '.join(prediction).replace('"', ''))
                translation = revert_BPE(sentence)
                file.write(str(translation))
