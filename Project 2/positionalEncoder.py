import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoder(nn.Module):
    def __init__(self, vocabulary_size, word_embedding_size, pos_embedding_size, max_length):
        super(PositionalEncoder, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = word_embedding_size + pos_embedding_size
        self.max_length = max_length
        self.word_embedding = nn.Embedding(vocabulary_size, word_embedding_size)
        self.pos_embedding = nn.Embedding(max_length, pos_embedding_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, sentence):
        encoder_output = torch.zeros(self.max_length, 1, self.hidden_size, device=device)
        seq_len = len(sentence)
        word_embedding = self.word_embedding(sentence).view(seq_len, 1, -1)
        positions = [i for i in range(0, seq_len)]
        position_tensors = Variable(torch.tensor(positions, device=device).view(-1, 1))
        pos_embedding = self.pos_embedding(position_tensors).view(seq_len, 1, -1)
        output = torch.cat((word_embedding, pos_embedding), dim=2)
        hidden = output.mean(dim=0).view(1, 1, -1)
        encoder_output[:seq_len, :, :] = output
        hidden = self.hidden_layer(hidden)

        return encoder_output, hidden

