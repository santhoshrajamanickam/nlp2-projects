from collections import defaultdict
import random
import math
import time
import numpy as np
# from scipy.special import digamma, psi
from aer import AERSufficientStatistics, read_naacl_alignments

class IBM:

    def __init__(self, model, corpus, limit=None):
        self.model = model

        self.t = defaultdict(lambda: 1/len(corpus.french_vocab)) # translation parameters
        l = 1
        self.q = defaultdict(lambda: (1/(l+1))) 

        self.english_sentences = corpus.training_english[:limit]
        self.french_sentences = corpus.training_french[:limit]
        self.testing_english = corpus.testing_english
        self.testing_french = corpus.testing_french

    def run_epoch(self, S, approach):

        print("===========")
        print("Starting {}".format(approach))
        print("===========")

        training_size = len(self.english_sentences)

        for s in range(S):
            word_counts = defaultdict(lambda: 0)
            english_word_counts = defaultdict(lambda: 0)

            log_likelihood = 0.0
            start = time.time()

            for k in range(training_size):

                l = len(self.english_sentences[k])
                m = len(self.french_sentences[k])
                total_english_counts = defaultdict(lambda: 0)

                for i in range(0, m):

                    french_word = self.french_sentences[k][i]

                    for j in range(0, l):
                        total_english_counts[self.english_sentences[k][j]] += self.t[(french_word, self.english_sentences[k][j])]

                    for j in range(0, l):
                        english_word = self.english_sentences[k][j]
                        word_counts[(french_word, english_word)] += self.t[(french_word, english_word)]/total_english_counts[english_word]
                        english_word_counts[english_word] += self.t[(french_word, self.english_sentences[k][j])]/total_english_counts[english_word]

            for keys in word_counts.keys():
                self.t[(keys[0], keys[1])] = word_counts[(keys[0], keys[1])]/english_word_counts[keys[1]]
                log_likelihood += math.log(self.t[(keys[0], keys[1])])

            time_taken = (time.time() - start)
            print("Iteration {}: took {} secs (Log-likelihood: {})".format(s, time_taken, log_likelihood))

    def viterbi_alignment(self):

        testing_size = len(self.testing_english)
        test_alignments = []

        for k in range(testing_size):

            l = len(self.testing_english[k])
            m = len(self.testing_french[k])

            alignment = set()

            for i in range(0,m):
                all_alignments = []
                for j in range(0,l):
                    all_alignments.append(self.t[(self.testing_french[k][i], self.testing_english[k][j])] * self.q[(j, i+1, l, m)])
                alignment.add((all_alignments.index(max(all_alignments)),i+1))
            test_alignments.append(alignment)

        return test_alignments

    def calculate_aer(self, eval_alignement_path, test_alignments):

        gold_standard = read_naacl_alignments(eval_alignement_path)

        metric = AERSufficientStatistics()

        for gold_alignments, test_alignments in zip(gold_standard, test_alignments):
            metric.update(sure=gold_alignments[0], probable=gold_alignments[1], predicted=test_alignments)

        aer = metric.aer()

        print("AER: {}".format(aer))
