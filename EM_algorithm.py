from collections import defaultdict
import random
import math
import time

def EM_algorithm(english_sentences, french_sentences):

    t = defaultdict(lambda: random.uniform(0, 1)) # translation parameters
    q = defaultdict(lambda: random.uniform(0, 1)) # distortion/alignment parameters

    S = 10 # number of iterations
    training_size = len(english_sentences)
    previous_likelihood = 0.0

    for s in range(S):
        word_counts = defaultdict(lambda: 0)
        english_word_counts = defaultdict(lambda: 0)
        alignment_counts  = defaultdict(lambda: 0)
        french_alignment_counts = defaultdict(lambda: 0)

        log_likelihood = 0.0
        start = time.time()

        for k in range(training_size):
            for i in range(1,len(french_sentences[k])):

                french_word = french_sentences[k][i-1]
                l = len(english_sentences[k])
                m = len(french_sentences[k])
                normalization_constant = 0.0
                precompute_delta = []

                for index in range(0,len(english_sentences[k])):
                    precompute_delta.append(float(q[(index, i+1, l, m)]*t[(french_word, english_sentences[k][index])]))
                normalization_constant = float(sum(precompute_delta))
                log_likelihood += math.log(normalization_constant)

                for j in range(0,len(english_sentences[k])):
                    english_word = english_sentences[k][j]
                    # print(j,len(precompute_delta))
                    delta = precompute_delta[j]/normalization_constant

                    word_counts[(english_word, french_word)] +=  delta
                    english_word_counts[english_word] += delta
                    alignment_counts[(j, i+1, l, m)] += delta
                    french_alignment_counts[(i+1, l, m)] += delta

            # print("Done sentence {}.".format(k))
        for keys in word_counts.keys():
            t[(keys[1], keys[0])] = word_counts[(keys[0], keys[1])]/english_word_counts[keys[0]]

        for keys in alignment_counts.keys():
            q[(keys[0], keys[1], keys[2], keys[3])] = alignment_counts[(keys[0], keys[1], keys[2], keys[3])]/french_alignment_counts[keys[1], keys[2], keys[3]]

        time_taken = (time.time() - start)
        print("Iteration {}: took {} secs (Log-likelihood: {})".format(s, time_taken, log_likelihood))

        if (previous_likelihood == log_likelihood):
            break
        else:
            previous_likelihood = log_likelihood
        # print(t)
        # print(q)

    return t, q
