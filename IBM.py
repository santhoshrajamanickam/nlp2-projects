from collections import defaultdict
import random
import math
import time
import numpy as np
from scipy.special import digamma, psi
from aer import AERSufficientStatistics, read_naacl_alignments

class IBM:

    def __init__(self, model, training_english, training_french, testing_english=None, testing_french=None, alpha=0.02):
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

    def em_algorithm(self, S):

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

                # print("Done sentence {}.".format(k))
            for keys in word_counts.keys():
                self.t[(keys[1], keys[0])] = word_counts[(keys[0], keys[1])]/english_word_counts[keys[0]]

            for keys in alignment_counts.keys():
                self.q[(keys[0], keys[1], keys[2], keys[3])] = alignment_counts[(keys[0], keys[1], keys[2], keys[3])]/french_alignment_counts[keys[1], keys[2], keys[3]]

            time_taken = (time.time() - start)
            print("Iteration {}: took {} secs (Log-likelihood: {})".format(s, time_taken, log_likelihood))

            if (previous_likelihood == log_likelihood):
                break
            else:
                previous_likelihood = log_likelihood
            self.likelihoods.append(log_likelihood)

        self.final_likelihood = log_likelihood


    def var_inference(self, S):

        print("===========")
        print("Starting Variational Inference")
        print("===========")


        training_size = len(self.english_sentences)
        for s in range(S):

            start = time.time()

            word_counts = defaultdict(lambda: 0)
            english_word_counts = defaultdict(lambda: 0)
            nr_of_english_word_counts = defaultdict(lambda: 0)
            alignment_counts  = defaultdict(lambda: 0)
            french_alignment_counts = defaultdict(lambda: 0)


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

                    for j in range(0, l):
                        english_word = self.english_sentences[k][j]
                        delta = precompute_delta[j]/normalization_constant

                        word_counts[(english_word, french_word)] +=  delta
                        english_word_counts[english_word] += delta
                        nr_of_english_word_counts[english_word] += 1
                        alignment_counts[(j, i+1, l, m)] += delta
                        french_alignment_counts[(i+1, l, m)] += delta


            for keys, count in word_counts.items():
                count_all_fe = english_word_counts[keys[0]]
                nr_of_f = nr_of_english_word_counts[keys[0]]
                self.t[(keys[1], keys[0])] = psi(count + self.alpha) - psi(count_all_fe + self.alpha * nr_of_f)

            for keys in alignment_counts.keys():
                self.q[(keys[0], keys[1], keys[2], keys[3])] = alignment_counts[(keys[0], keys[1], keys[2], keys[3])]/french_alignment_counts[keys[1], keys[2], keys[3]]



            time_taken = (time.time() - start)
            print("Iteration {}: took {} secs (ELBO: )".format(s, time_taken))


    def compute_elbo(self, mle, lambdas_fe, lambda_sum):
        kl = 0
        # TODO: fix for current approach without vocabulary size
        for e in range(english_vocabulary_size):
            for f in range(french_vocabulary_size):
                kl_sum = (digamma(t[f,e]) - digamma(sum(t[:,e]) - t[f,e])) * (self.alpha - t[f,e]) + loggamma(t[f,e]) - loggamma(self.alpha)
                kl_sum += loggamma(self.alpha*F_vocab_size) - loggamma(sum(t[:,e]))
                kl_e = kl_sum.real
            kl += kl_e

        return (-kl + mle)

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
