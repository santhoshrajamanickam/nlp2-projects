# all functions are from the seq2seq tutorial on pytorch.org
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(english, french, pair):
    input_tensor = tensorFromSentence(english, pair[0])
    target_tensor = tensorFromSentence(french, pair[1])
    return (input_tensor, target_tensor)
