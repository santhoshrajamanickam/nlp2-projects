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

def evaluate(en, fr, encoder, decoder, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(en, sentence)
        input_length = input_tensor.size()[0]

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_output, encoder_hidden = encoder(input_tensor)

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

def evaluateRandomly(fr, en, encoder, decoder, pairs, max_length, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(fr, en, encoder, decoder, pair[0], max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



def BLUE(translation, reference, outputfile):
    os.system('multi-bleu.perl ' + reference + ' < ' + translation + ' > ' + outputfile)
