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


# use Lang datastructure from Pytorch seq2seq tutorial
english = Lang(english_data)
french = Lang(french_data)

# create parallel sentence pairs
sentences = list(zip(english_data, french_data))
for sent in sentences:
    en_sent = sent[0].split(' ')
    fr_sent = sent[1].split(' ')
    english.addSentence(en_sent)
    french.addSentence(fr_sent)

# example print of revert BPE
test = sentences[1]
print(test[0])
print(revert_BPE(test[0]))


input_voc_size = english.n_words
output_voc_size = french.n_words

print(input_voc_size)


encoder = PositionalEncoder(input_voc_size, word_embedding_size, pos_embedding_size, maximum_length)
decoder = AttnDecoderRNN(hidden_size, output_voc_size, maximum_length)

print('===============Training model...====================')
n_iters = 5000
epoch(english, french, sentences, encoder, decoder, n_iters, maximum_length, 500)



print('===============Calculating metrics...===================')
sentences = load_data('data/test/test_2017_flickr.en')
with open('translations.txt', 'w') as file:
    for sent_pair in sentences:
        prediction, _ = evaluate(english, french, encoder, decoder, sent_pair[0], maximum_length)
        sentence = (' '.join(prediction).replace('"',''))
        translation = revert_BPE(sentence)
        file.write(translation)
        file.write('\n')

os.system("perl multi-bleu.pl -lc data/test/test_2017_flickr.fr < test_predictions.txt")
