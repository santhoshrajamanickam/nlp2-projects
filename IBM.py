from collections import defaultdict
import math
import random
import time
import numpy as np
from scipy.special import digamma, gammaln
from aer import AERSufficientStatistics, read_naacl_alignments


class IBM:

    def __init__(self, model, corpus, limit=None, initialization='random', pretrained_t=None):
        self.model = model
        self.initialization = initialization

        if self.model == 'IBM1':
            self.t = defaultdict(lambda: 1 / len(corpus.french_vocab))  # translation parameters
            l = 1
            self.q = defaultdict(lambda: (1 / (l + 1)))  # distortion/alignment parameters
        elif self.model == 'IBM2':
            self.max_jump = 100
            if self.initialization == 'uniform':
                self.t = defaultdict(lambda: 1 / len(corpus.french_vocab))  # translation parameters
                l = 1
                self.q = defaultdict(lambda: (1 / (l + 1)))  # distortion/alignment parameters
            elif self.initialization == 'random':
                self.t = defaultdict(lambda: random.uniform(0, 1))  # translation parameters
                self.q = 1. / (2 * self.max_jump) * np.ones((1, 2 * self.max_jump))
            elif self.initialization == 'IBM1':
                self.t = pretrained_t  # translation parameters
                self.q = 1. / (2 * self.max_jump) * np.ones((1, 2 * self.max_jump))  # distortion/alignment parameters

        self.log_likelihood = 0
        self.elbo = 0

        self.english_sentences = corpus.training_english[:limit]
        self.french_sentences = corpus.training_french[:limit]
        self.testing_english = corpus.testing_english
        self.testing_french = corpus.testing_french

    def jump(self, i, j, m, n):
        max_jump = 99
        jump = np.floor(i - (j * m / n))
        if jump > max_jump:
            jump = max_jump
        elif jump < -max_jump:
            jump = -max_jump
        return jump + max_jump

    def run_epoch(self, S, approach, alpha=None):

        print("===========")
        print("Starting {}".format(approach))
        print("===========")

        training_size = len(self.english_sentences)

        for s in range(S):
            if approach == 'EM':
                word_counts = defaultdict(lambda: 0)
                english_word_counts = defaultdict(lambda: 0)
            else:
                sum_lambdas = defaultdict(lambda: defaultdict (lambda: alpha))
                # sum_lambdas = defaultdict(lambda: alpha)
                elbo = 0.0

            if self.model == 'IBM2':
                jump_counts = np.zeros((1, 2 * self.max_jump), dtype=np.float)

            log_likelihood = 0.0
            start = time.time()

            for k in range(training_size):

                l = len(self.english_sentences[k])
                m = len(self.french_sentences[k])

                if self.model == 'IBM1':
                    total_french_counts = defaultdict(lambda: 0)

                    for i in range(0, m):
                        french_word = self.french_sentences[k][i]
                        total_french_counts[french_word] = 0
                        for j in range(0, l):
                            total_french_counts[french_word] += self.t[
                                (french_word, self.english_sentences[k][j])]

                for i in range(0, m):
                    french_word = self.french_sentences[k][i]
                    precompute_delta = []

                    if self.model == 'IBM1':
                        for index in range(0, l):
                            precompute_delta.append(float(
                                self.q[(index, i + 1, l, m)] * self.t[(french_word, self.english_sentences[k][index])]))
                    else:
                        for index in range(0, l):
                            jump = self.jump(index, i, l, m)
                            precompute_delta.append(
                                float(self.q[0, int(jump)] * self.t[(french_word, self.english_sentences[k][index])]))

                    normalization_constant = float(sum(precompute_delta))
                    log_likelihood += math.log(normalization_constant)

                    if self.model == 'IBM1':
                        for j in range(0, l):
                            english_word = self.english_sentences[k][j]
                            if approach == 'EM':
                                delta = self.t[(french_word, english_word)] / total_french_counts[french_word]
                                word_counts[(french_word, english_word)] += delta
                                english_word_counts[english_word] += delta
                            else:
                                delta = self.t[(french_word, english_word)] / total_french_counts[french_word]
                                # lambdas[french_word][english_word] += delta
                                sum_lambdas[english_word][french_word] += delta
                    if self.model == 'IBM2':
                        for j in range(0, l):
                            english_word = self.english_sentences[k][j]
                            delta = precompute_delta[j] / normalization_constant
                            word_counts[(french_word, english_word)] += delta
                            english_word_counts[english_word] += delta
                            jump_counts[0, int(self.jump(j, i, l, m))] += delta

            if approach == 'EM':
                for keys in word_counts.keys():
                    self.t[(keys[0], keys[1])] = word_counts[(keys[0], keys[1])]/english_word_counts[keys[1]]
            else:
                for english_word in sum_lambdas.keys():
                    target_norm = digamma(sum(sum_lambdas[english_word].values()))
                    for french_word in sum_lambdas[english_word].keys():
                        self.t[(french_word, english_word)] = math.exp(
                            digamma(sum_lambdas[english_word][french_word]) - target_norm)


                # for french_word in lambdas.keys():
                #     for english_word in lambdas[french_word].keys():
                #         self.t[(french_word, english_word)] = math.exp(
                #             digamma(lambdas[french_word][english_word])
                #             - digamma(sum_lambdas[english_word]))

                # elbo += log_likelihood
                # sum_lambda = 0
                # sum_alpha = 0
                kl_sum = 0

                for english_word in sum_lambdas.keys():

                    theta_sum = 0
                    sum_lambda = 0
                    sum_alpha = 0
                    target_norm = digamma(sum(sum_lambdas[english_word].values()))

                    for french_word in sum_lambdas[english_word].keys():
                        sufficient_statistic = digamma(sum_lambdas[english_word][french_word]) - target_norm
                        theta_sum += sufficient_statistic * \
                                     (alpha - sum_lambdas[english_word][french_word]) + \
                                     gammaln(sum_lambdas[english_word][french_word]) - gammaln(alpha)
                        sum_alpha += alpha
                        sum_lambda += sum_lambdas[english_word][french_word]

                    kl_divergence = theta_sum + gammaln(sum_alpha) + gammaln(sum_lambda)
                    kl_sum += kl_divergence

                self.elbo = log_likelihood + kl_sum


                # for french_word in lambdas.keys():
                #     for english_word in lambdas[french_word].keys():
                #         # print(lambdas[french_word][english_word])
                #         kl_sum += math.log(self.t[(french_word,english_word)])*(alpha - lambdas[french_word][english_word]) \
                #                 + loggamma(lambdas[french_word][english_word]) - loggamma(alpha)
                #         sum_lambda += loggamma(lambdas[french_word][english_word])
                #         sum_alpha += loggamma(alpha)
                #
                # kl_sum += sum_alpha
                # kl_sum -= sum_lambda
                #
                # elbo = log_likelihood - kl_sum.real

            if self.model == 'IBM2':
                self.q = 1. / float(np.sum(jump_counts)) * jump_counts
                
            time_taken = (time.time() - start)
            if approach == 'EM':
                print("Iteration {}: took {} secs (Log-likelihood: {})".format(s, time_taken, log_likelihood))
            else:
                print("Iteration {}: took {} secs (ELBO: {})".format(s, time_taken, self.elbo))

        return self.t

    def viterbi_alignment(self):

        testing_size = len(self.testing_english)
        test_alignments = []

        for k in range(testing_size):

            l = len(self.testing_english[k])
            m = len(self.testing_french[k])

            alignment = set()

            if self.model == 'IBM1':
                for i in range(0, m):
                    all_alignments = []
                    for j in range(0, l):
                        all_alignments.append(
                            self.t[(self.testing_french[k][i], self.testing_english[k][j])] * self.q[(j, i + 1, l, m)])
                    alignment.add((all_alignments.index(max(all_alignments)), i + 1))
                test_alignments.append(alignment)
            else:
                for i in range(0, m):
                    all_alignments = []
                    for j in range(0, l):
                        jump = self.jump(j, i, l, m)
                        all_alignments.append(
                            self.t[(self.testing_french[k][i], self.testing_english[k][j])] * self.q[0, int(jump)])
                    alignment.add((all_alignments.index(max(all_alignments)), i + 1))
                test_alignments.append(alignment)

        return test_alignments

    def calculate_aer(self, eval_alignement_path, test_alignments):

        gold_standard = read_naacl_alignments(eval_alignement_path)

        metric = AERSufficientStatistics()

        for gold_alignments, test_alignments in zip(gold_standard, test_alignments):
            metric.update(sure=gold_alignments[0], probable=gold_alignments[1], predicted=test_alignments)

        aer = metric.aer()

        print("AER: {}".format(aer))