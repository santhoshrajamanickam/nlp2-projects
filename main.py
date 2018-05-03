from data_load import ParallelCorpus
from IBM import IBM
# from IBM2 import IBM2
# import matplotlib.pyplot as plt

english_training_filepath = "./training/hansards.36.2.e"
french_training_filepath = "./training/hansards.36.2.f"
english_testing_filepath = "./validation/dev.e"
french_testing_filepath = "./validation/dev.f"
gold_standard_filepath = "./validation/dev.wa.nonullalign"
num_iteration = 10

corpus = ParallelCorpus(english_training_filepath, french_training_filepath, \
                        english_testing_filepath, french_testing_filepath)

# print(len(corpus.english_vocab))
# print(len(corpus.french_vocab))

IBM_model_1 = IBM('IBM1', corpus, limit=10000)
IBM_model_1.run_epoch(num_iteration,'EM')
test_alignments = IBM_model_1.viterbi_alignment()
IBM_model_1.calculate_aer(gold_standard_filepath, test_alignments)

# IBM_model_2 = IBM2(corpus.training_english[:1000], corpus.training_french[:1000])
# IBM_model_2.load_test_sentences(corpus.testing_english, corpus.testing_french)
# IBM_model_2.run_epoch(num_iteration)
# test_alignments = IBM_model_2.viterbi_alignment()
# IBM_model_2.calculate_aer(gold_standard_filepath, test_alignments)

#plt.plot(range(len(IBM_model_1.likelihoods)), IBM_model_1.likelihoods)
#plt.show()

# IBM_model_2 = IBM(model, corpus.training_english, corpus.training_french)
# IBM_model_2.load_test_sentences(corpus.testing_english, corpus.testing_french)
# IBM_model_2.em_algorithm(num_iteration)
# test_alignments = IBM_model_2.viterbi_alignment()
# IBM_model_2.calculate_aer(gold_standard_filepath, test_alignments)

# t, q = EM_algorithm(corpus.training_english[:1000], corpus.training_french[:1000])
