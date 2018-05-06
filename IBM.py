from collections import defaultdict
import math
import time
from scipy.special import digamma, psi
from aer import AERSufficientStatistics, read_naacl_alignments


class IBM:

    def __init__(self, model, corpus, limit=None):
        self.model = model

        self.t = defaultdict(lambda: 1/len(corpus.french_vocab))  # translation parameters
        l = 1
        self.q = defaultdict(lambda: (1/(l+1)))  # distortion/alignment parameters

        self.english_sentences = corpus.training_english[:limit]
        self.french_sentences = corpus.training_french[:limit]
        self.testing_english = corpus.testing_english
        self.testing_french = corpus.testing_french

    def run_epoch(self, S, approach, alpha=None):

        print("===========")
        print("Starting {}".format(approach))
        print("===========")

        training_size = len(self.english_sentences)

        for s in range(S):
            word_counts = defaultdict(lambda: 0)
            english_word_counts = defaultdict(lambda: 0)
            lambdas = defaultdict(lambda: defaultdict (lambda: alpha))

            log_likelihood = 0.0
            start = time.time()

            for k in range(training_size):

                l = len(self.english_sentences[k])
                m = len(self.french_sentences[k])

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

                    for index in range(0, l):
                        precompute_delta.append(float(
                            self.q[(index, i + 1, l, m)] * self.t[(french_word, self.english_sentences[k][index])]))

                    normalization_constant = float(sum(precompute_delta))
                    log_likelihood += math.log(normalization_constant)

                    for j in range(0, l):
                        english_word = self.english_sentences[k][j]
                        if approach == 'EM':
                            delta = self.t[(french_word, english_word)] / total_french_counts[french_word]
                            word_counts[(french_word, english_word)] += delta
                            english_word_counts[english_word] += delta
                        else:
                            delta = self.t[(french_word, english_word)] / total_french_counts[french_word]
                            lambdas[french_word][english_word] += delta
                            english_word_counts[english_word] += delta

            if approach == 'EM':
                for keys in word_counts.keys():
                    self.t[(keys[0], keys[1])] = word_counts[(keys[0], keys[1])]/english_word_counts[keys[1]]
            else:

                for french_word in lambdas.keys():
                    for english_word in lambdas[french_word].keys():
                        # english_word_sum = 1
                        # for other_french_word in lambdas.keys():
                        #     if other_french_word != french_word:
                        #         english_word_sum += lambdas[other_french_word][english_word] + alpha
                        self.t[(french_word, english_word)] = math.exp(
                            digamma(lambdas[french_word][english_word]+ alpha)
                            - digamma(english_word_counts[english_word]+alpha))

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
