# all functions are from the seq2seq tutorial on pytorch.org
import time
import math
import torch
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1


# Return a list of indexes, one for each word in the sentence
def indexesFromSentence(lang, sentence):
    indices = []
    for word in sentence.split(' '):
        try:
            indices.append(lang.word2index[word])
        except KeyError:
            indices.append(lang.word2index['UNK'])
    return indices


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(french, english, pair):
    input_tensor = tensorFromSentence(french, pair[0])
    target_tensor = tensorFromSentence(english, pair[1])
    return input_tensor, target_tensor


def variable_from_sentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1))
    return var


def variables_from_pair(input_lang, output_lang, pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return input_variable, target_variable

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

