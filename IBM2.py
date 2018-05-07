from collections import defaultdict
import random
import math
import time
import numpy as np
from aer import AERSufficientStatistics, read_naacl_alignments

class IBM2:
    def __init__(self, training_english, training_french, vocab_size, testing_english=None, testing_french=None, initialization=None, prelearned=None):
        if initialization == 'uniform':
            self.t = 1.0/vocab_size
        elif initialization == 'ibm1':
            if prelearned == None:
                raise Exception('No pretrained translation parameters are given!')
            self.t = prelearned
        else:
            self.t = defaultdict(lambda: random.uniform(0, 1)) # translation parameters, default random
        self.q = defaultdict(lambda: random.uniform(0, 1)) # distortion/alignment parameters
        self.english_sentences = training_english
        self.french_sentences = training_french
        self.testing_english = testing_english
        self.testing_french = testing_french
        self.likelihoods = []
        self.max_jump = 100
        self.jump_dist = 1. / (2 * self.max_jump) * np.ones((1, 2 * self.max_jump))


    def load_test_sentences(self, testing_english, testing_french):
        self.testing_english = testing_english
        self.testing_french = testing_french

    def get_voc_size(self):
        english_training = TextData()


    def jump(self,i, j, m, n):
        max_jump = 99
        jump = np.floor(i - (j * m / n))
        if jump > max_jump:
            jump = max_jump
        elif jump < -max_jump:
            jump = -max_jump
        return jump + max_jump

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
                        q = self.jump(index,i,l,m)
                        precompute_delta.append(float(self.jump_dist[0,int(q)]*self.t[(french_word, self.english_sentences[k][index])]))

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
                    q = self.jump(j,i,l,m)
                    all_alignments.append(self.t[(self.testing_french[k][i], self.testing_english[k][j])] * self.jump_dist[0,int(q)])
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

    def calculate_perplexity(self):
        perplexity = 0
        training_size = len(self.english_sentences)
        for k in range(training_size):
            m = len(self.french_sentences[k])
            l = len(self.english_sentences[k])
            total_prob = 1
            for i in range(m):
                prob = 0
                french_word = self.french_sentences[k][i]
                for j in range(l):
                    english_word = self.english_sentences[k][j]
                    prob += self.t[(french_word, english_word)] * jump(j,i,l,m)
                total_prob *= prob
            perplexity += np.log2(total_prob)
        return perplexity
