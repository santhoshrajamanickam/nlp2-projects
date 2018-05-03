from collections import defaultdict
import random
import math
import time
import numpy as np
from scipy.special import digamma, psi
from aer import AERSufficientStatistics, read_naacl_alignments

class IBM:

    def __init__(self, model, training_english, training_french, testing_english=None, testing_french=None, alpha=0.001):
        self.model = model
        self.t = defaultdict(lambda: random.uniform(0, 1)) # translation parameters
        if self.model == 2:
            self.q = defaultdict(lambda: random.uniform(0, 1)) # distortion/alignment parameters
        else:
            l = 1
            self.q = defaultdict(lambda: (1/(l+1)))
        self.english_sentences = training_english
        self.french_sentences = training_french
        self.testing_english = testing_english
        self.testing_french = testing_french
        self.likelihoods = []
        self.alpha = alpha # prior value for VI IBM1

    def load_test_sentences(self, testing_english, testing_french):
        self.testing_english = testing_english
        self.testing_french = testing_french

    def run_epoch(self, S, approach):

        print("===========")
        print("Starting {}".format(approach))
        print("===========")

        training_size = len(self.english_sentences)
        previous_likelihood = 0.0

        for s in range(S):
            word_counts = defaultdict(lambda: 0)
            english_word_counts = defaultdict(lambda: 0)
            nr_of_english_word_counts = defaultdict(lambda: 0)
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
                    if approach == 'EM':
                        log_likelihood += math.log(normalization_constant)

                    for j in range(0, l):
                        english_word = self.english_sentences[k][j]
                        delta = precompute_delta[j]/normalization_constant
                        if approach == 'EM':
                            word_counts[(english_word, french_word)] +=  delta
                            english_word_counts[english_word] += delta
                        if approach == 'VI':
                            word_counts[(english_word, french_word)] += precompute_delta[j]
                            english_word_counts[english_word] += precompute_delta[j]

                        # total count is only used for VI
                        nr_of_english_word_counts[english_word] += 1
                        alignment_counts[(j, i+1, l, m)] += delta
                        french_alignment_counts[(i+1, l, m)] += delta

            if approach == 'VI':
                # for keys, count in word_counts.items():
                #     count_all_fe = english_word_counts[keys[0]]
                #     nr_of_f = nr_of_english_word_counts[keys[0]]
                #     self.t[(keys[1], keys[0])] = psi(count + self.alpha) - psi(count_all_fe + self.alpha * nr_of_f)
                for keys, count in word_counts.items():
                    self.t[(keys[1], self.english_sentences[k][j])] = math.exp(psi(self.alpha + word_counts[(keys[0], keys[1])]) - psi(english_word_counts[keys[0]]))

            # if approach == 'EM':
            #     for keys in word_counts.keys():
            #         self.t[(keys[1], keys[0])] = word_counts[(keys[0], keys[1])]/english_word_counts[keys[0]]
            #
            #     if (previous_likelihood == log_likelihood):
            #         break
            #     else:
            #         previous_likelihood = log_likelihood
            #     self.likelihoods.append(log_likelihood)




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
