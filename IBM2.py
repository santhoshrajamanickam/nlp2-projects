from collections import defaultdict
import random
import math
import time
import numpy as np
from aer import AERSufficientStatistics, read_naacl_alignments

class IBM2:
    def __init__(self, training_english, training_french, testing_english=None, testing_french=None):
        self.t = defaultdict(lambda: random.uniform(0, 1)) # translation parameters
        self.q = defaultdict(lambda: random.uniform(0, 1)) # distortion/alignment parameters
        self.english_sentences = training_english
        self.french_sentences = training_french
        self.testing_english = testing_english
        self.testing_french = testing_french
        self.likelihoods = []

    def load_test_sentences(self, testing_english, testing_french):
        self.testing_english = testing_english
        self.testing_french = testing_french

    def run_epoch(self, S):

        print("===========")
        print("Starting EM")
        print("===========")

        training_size = len(self.english_sentences)
        previous_likelihood = 0.0

        for s in range(S):
            word_counts = defaultdict(lambda: 0)
            english_word_counts = defaultdict(lambda: 0)
            alignment_counts  = defaultdict(lambda: 0)
            french_alignment_counts = defaultdict(lambda: 0)

            log_likelihood = 0.0
            start = time.time()

            for k in range(training_size):

                l = len(self.english_sentences[k])
                m = len(self.french_sentences[k])

                for i in range(0, m):

                    french_word = self.french_sentences[k][i]

                    normalization_constant = 0.0
                    precompute_delta = []

                    for index in range(0, l):
                        precompute_delta.append(float(self.q[(index, i+1, l, m)]*self.t[(french_word, self.english_sentences[k][index])]))

                    normalization_constant = float(sum(precompute_delta))
                    log_likelihood += math.log(normalization_constant)

                    for j in range(0, l):
                        english_word = self.english_sentences[k][j]
                        delta = precompute_delta[j]/normalization_constant
                        word_counts[(english_word, french_word)] +=  delta
                        english_word_counts[english_word] += delta

                        alignment_counts[(j, i+1, l, m)] += delta
                        french_alignment_counts[(i+1, l, m)] += delta

            for keys in word_counts.keys():
                self.t[(keys[1], keys[0])] = word_counts[(keys[0], keys[1])]/english_word_counts[keys[0]]


            for keys in alignment_counts.keys():
                self.q[(keys[0], keys[1], keys[2], keys[3])] = alignment_counts[(keys[0], keys[1], keys[2], keys[3])]/french_alignment_counts[keys[1], keys[2], keys[3]]

            time_taken = (time.time() - start)
            print("Iteration {}: took {} secs (Log-likelihood: {})".format(s, time_taken, log_likelihood))

        self.final_likelihood = log_likelihood





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
