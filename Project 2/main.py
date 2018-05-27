from positionalModel import *
from rnnModel import *
from lang import Lang
from evaluate import *
from data_process import load_data, revert_BPE
import nltk as nltk
import os


# hyperparameters
word_embedding_size = 512
pos_embedding_size = 50
#hidden_size = word_embedding_size + pos_embedding_size
hidden_size = word_embedding_size
maximum_length = 50

english_data = load_data('data/train/train_complete.en')
french_data = load_data('data/train/train_complete.fr')

print("Nr of English sentences: ", len(english_data))
print("Nr of French sentences: ", len(french_data))

# use Lang datastructure from Pytorch seq2seq tutorial
english = Lang(english_data)
french = Lang(french_data)

# create parallel sentence pairs
sentences = list(zip(french_data, english_data))
for sent in sentences:
    fr_sent = sent[0].split(' ')
    en_sent = sent[1].split(' ')
    english.addSentence(en_sent)
    french.addSentence(fr_sent)
english.addWord('UNK')
french.addWord('UNK')

# example print of revert BPE
test = sentences[1]
print(test[0])
print(revert_BPE(test[0]))


input_voc_size = french.n_words
output_voc_size = english.n_words

print("Size of English vocabulary: ", output_voc_size)
print("Size of French vocabulary: ", input_voc_size)


#encoder = PositionalEncoder(input_voc_size, word_embedding_size, pos_embedding_size, maximum_length)
encoder = EncoderRNN(input_voc_size, word_embedding_size)
decoder = AttnDecoderRNN(word_embedding_size, output_voc_size, maximum_length)


print('===============Training model...====================')
# n_iters = 8
# epochRNN(french, english, sentences, encoder, decoder, n_iters, maximum_length)
#
# # save models
# torch.save(encoder.state_dict(), 'models/POSencoder_final')
# torch.save(decoder.state_dict(), 'models/POSdecoder_final')


encoder.load_state_dict(torch.load('models/RNNencoder_3'))
decoder.load_state_dict(torch.load('models/RNNdecoder_3'))

with open('POSloss', 'rb') as handle:
    loss = pickle.load(handle)
print(loss)

# print('===============Calculating metrics...===================')
sentences = load_data('data/test/test_complete.fr')
translations = []
with open('test_predictions.txt', 'w') as file:
    for i, sent in enumerate(sentences):
        #prediction, _ = evaluate(french, english, encoder, decoder, sent, maximum_length)
        prediction, _ = evaluateRNN(french, english, encoder, decoder, sent, maximum_length)
        #evaluateAndShowAttention(french, english, encoder, decoder, sent, maximum_length)
        sentence = (' '.join(prediction).replace('"',''))
        translation = revert_BPE(sentence)
        translations.append(translation)
        file.write(str(translation))

os.system("perl multi-bleu.perl -lc data/test/test_lower.en < test_predictions.txt")
os.system("java -Xmx2G -jar meteor-1.5/meteor-*.jar test_predictions.txt data/test/test_lower.en -l en -norm -a meteor-1.5/data/paraphrase-en.gz")
