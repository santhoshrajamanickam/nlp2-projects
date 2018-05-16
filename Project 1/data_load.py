# import pandas as pd
from collections import Counter

class ParallelCorpus:

    def __init__(self, english_training_path, french_training_path, english_testing_path, french_testing_path):

        self.training_english = []
        self.training_french = []
        self.testing_english = []
        self.testing_french = []
        self.english_vocab = set()
        self.french_vocab = set()

        print("============")
        print("Loading Data")
        print("============")

        with open(english_training_path, encoding='utf8') as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                self.english_vocab.update(sentence)
                sentence.insert(0,'NULL') # adding a NULL word
                self.training_english.append(sentence)
            #self.training_english = self.map_to_unk(20,self.training_english)
        self.english_vocab.update('NULL')

        with open(french_training_path, encoding='utf8') as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                self.french_vocab.update(sentence)
                self.training_french.append(sentence)
            #self.training_french = self.map_to_unk(20,self.training_french)

        with open(english_testing_path, encoding='utf8') as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                sentence.insert(0,'NULL') # adding a NULL word
                self.testing_english.append(sentence)

        with open(french_testing_path, encoding='utf8') as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                self.testing_french.append(sentence)

        print("Number of English sentences in the training set: {}".format(len(self.training_english)))
        print("Number of French sentences in the training set: {}".format(len(self.training_french)))
        print("Number of English sentences in the testing set: {}".format(len(self.testing_english)))
        print("Number of French sentences in the testing set: {}".format(len(self.testing_french)))


    def map_to_unk(self, k, training):
        counts = Counter(w for sent in training for w in sent)
        counted_once = [w for w, count in counts.items() if count == 1]
        counted_once = counted_once[0:k]
        for i, sentence in enumerate(training):
            for j, word in enumerate(sentence):
                if word in counted_once:
                    sentence[j] = 'UNK'
            training[i] = sentence
        return training
