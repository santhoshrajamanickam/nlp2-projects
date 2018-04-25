# import pandas as pd
# from collections import defaultdict

class ParallelCorpus:

    def __init__(self, english_training_path, french_training_path, english_testing_path, french_testing_path):

        self.training_english = []
        self.training_french = []
        self.testing_english = []
        self.testing_french = []

        print("============")
        print("Loading Data")
        print("============")

        with open(english_training_path) as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                sentence.insert(0,'NULL') # adding a NULL word
                self.training_english.append(sentence)

        with open(french_training_path) as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                self.training_french.append(sentence)

        with open(english_testing_path) as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                sentence.insert(0,'NULL') # adding a NULL word
                self.testing_english.append(sentence)

        with open(french_testing_path) as file:
            for i, line in enumerate(file):
                sentence = [x.lower() for x in line.split()] # convert all words to lowercase
                self.testing_french.append(sentence)


        print("Number of English sentences in the training set: {}".format(len(self.training_english)))
        print("Number of French sentences in the training set: {}".format(len(self.training_french)))
        print("Number of English sentences in the testing set: {}".format(len(self.testing_english)))
        print("Number of French sentences in the testing set: {}".format(len(self.testing_french)))
