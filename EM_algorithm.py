from collections import defaultdict
import random

def EM_algorithm(english_sentences, french_sentences):

    t = defaultdict(lambda: defaultdict(lambda: random.uniform(0, 1))) # translation parameters
    q = defaultdict(lambda: defaultdict(lambda: defaultdict (lambda: random.uniform(0, 1)))) # distortion/alignment parameters

    S = 1 # number of iterations
    training_size = len(english_sentences)

    for s in range(S):
        word_counts = defaultdict(lambda: defaultdict(lambda: 0))
        english_word_counts = defaultdict(lambda: 0)
        alignment_counts  = defaultdict(lambda: defaultdict(lambda: defaultdict (lambda: 0)))
        french_alignment_counts = defaultdict(lambda: defaultdict (lambda: 0))
        for k in range(training_size):
            # print(french_sentences[k])
            # print(english_sentences[k])
            # print(*range(1,len(french_sentences[k]) + 1))
            # print(*range(0,len(english_sentences[k]) + 1))
            for i in range(1,len(french_sentences[k])):

                french_word = french_sentences[k][i]
                l = len(english_sentences[k])
                m = len(french_sentences[k])
                normalization_constant = 0

                for index in range(1,len(english_sentences[k])):
                    normalization_constant += q[index][i+1][(l,m)]*t[french_word][english_sentences[k][index]]

                for j in range(1,len(english_sentences[k])):
                    english_word = english_sentences[k][j]

                    delta = q[j][i+1][(l,m)]*t[french_word][english_word]/normalization_constant

                    word_counts[english_word][french_word] = word_counts[english_word][french_word] + delta
                    english_word_counts[english_word] = english_word_counts[english_word] + delta
                    alignment_counts[j][i+1][(l,m)] = alignment_counts[j][i+1][(l,m)] + delta
                    french_alignment_counts[i+1][(l,m)] = french_alignment_counts[i+1][(l,m)] + delta

            # print("Done sentence {}.".format(k))
        for french_word in t.keys():
            for english_word in t[french_word].keys():
                t[french_word][english_word] = word_counts[english_word][french_word]/english_word_counts[english_word]

        for english_index in q.keys():
            for french_index in q[english_index].keys():
                for lengths in q[english_index][french_index].keys():
                    q[english_index][french_index][lengths] = alignment_counts[english_index][french_index][lengths]/french_alignment_counts[french_index][lengths]

        print(t)
        print(q)

    return t, q
