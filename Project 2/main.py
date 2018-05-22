from positionalModel import *
from lang import Lang
from evaluate import *
from data_process import load_data, tokenize, revert_BPE
import nltk as nltk

# hyperparameters
word_embedding_size = 512
pos_embedding_size = 50
hidden_size = word_embedding_size + pos_embedding_size
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

# example print of revert BPE
test = sentences[1]
print(test[0])
print(revert_BPE(test[0]))


input_voc_size = french.n_words
output_voc_size = english.n_words

print("Size of English vocabulary: ", output_voc_size)
print("Size of French vocabulary: ", input_voc_size)


encoder = PositionalEncoder(input_voc_size, word_embedding_size, pos_embedding_size, maximum_length)
decoder = AttnDecoderRNN(hidden_size, output_voc_size, maximum_length)

print('===============Training model...====================')
n_iters = 100
epoch(french, english, sentences, encoder, decoder, n_iters, maximum_length, 500)



print('===============Calculating metrics...===================')
sentences = load_data('data/test/test_complete.fr')
with open('test_predictions.txt', 'w') as file:
    for sent in sentences:
        prediction, _ = evaluate(french, english, encoder, decoder, sent, maximum_length)
        sentence = (' '.join(prediction).replace('"',''))
        translation = revert_BPE(sentence)
        file.write(str(translation))

os.system("perl multi-bleu.pl -lc data/test/test_2017_flickr.en < test_predictions.txt")
