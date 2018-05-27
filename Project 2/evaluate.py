import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')
from helper import *
SOS_token = 0
EOS_token = 1

def evaluateRNN(en, fr, encoder, decoder, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(en, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(fr.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate(fr, en, encoder, decoder, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(fr, sentence)
        input_length = input_tensor.size()[0]
        if input_length > max_length:
            print('sentence too long')
            return
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        average_hidden = torch.zeros(1, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder((ei, input_tensor[ei]), encoder_hidden)
            average_hidden += encoder_output
            encoder_outputs[ei] += encoder_output[0, 0]
        encoder_hidden = (average_hidden/input_length).unsqueeze(0)


        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(en.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(fr, en, encoder, decoder, pairs, max_length, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(fr, en, encoder, decoder, pair[0], max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    attentions = attentions.numpy()
    cropped = []
    for a in attentions:
        cropped.append(a[38:50])
    cax = ax.matshow(cropped, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence[:100].split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('attention.png')


def evaluateAndShowAttention(fr, en, encoder, decoder, input_sentence, max_length):
    output_words, attentions = evaluate(
        fr, en, encoder, decoder, input_sentence, max_length)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
