# import pandas as pd
# from collections import defaultdict

class ParallelCorpus:

    def __init__(self, english_filepath, french_filepath):

        self.english_sentences = []
        self.french_sentences = []
        self.english2french = {}
        self.english_words = set()
        self.french_words = set()

        with open(english_filepath) as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                sentence.insert(0,'NULL')
                # self.english_words = self.english_words.union(sentence)
                self.english_sentences.append(sentence)

        with open(french_filepath) as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                # self.french_words = self.french_words.union(sentence)
                self.french_sentences.append(sentence)

        print("Number of English sentences: {}".format(len(self.english_sentences)))
        print("Number of French sentences: {}".format(len(self.french_sentences)))
        # print(len(self.english_words))
        # print(len(self.french_words))
        # print(self.english_sentences[5])
        # print(self.french_sentences[5])
