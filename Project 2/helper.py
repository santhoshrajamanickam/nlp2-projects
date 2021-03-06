# all functions are from the seq2seq tutorial on pytorch.org
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

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
    return (input_tensor, target_tensor)
