from data_load import ParallelCorpus
from IBM import IBM
from IBM2 import IBM2
from TextData import TextData
import matplotlib.pyplot as plt

english_training_filepath = "./training/hansards.36.2.e"
french_training_filepath = "./training/hansards.36.2.f"
english_testing_filepath = "./validation/dev.e"
french_testing_filepath = "./validation/dev.f"
gold_standard_filepath = "./validation/dev.wa.nonullalign"
num_iteration = 14

corpus = ParallelCorpus(english_training_filepath, french_training_filepath, \
                        english_testing_filepath, french_testing_filepath)

vocab_size = TextData("training/hansards.36.2.f").vocab_size

IBM_model_2 = IBM2(corpus.training_english[:10000], corpus.training_french[:10000], vocab_size, initialization='uniform')
IBM_model_2.load_test_sentences(corpus.testing_english, corpus.testing_french)
IBM_model_2.run_epoch(num_iteration)
test_alignments = IBM_model_2.viterbi_alignment()
IBM_model_2.calculate_aer(gold_standard_filepath, test_alignments)

#plt.plot(range(len(IBM_model_1.likelihoods)), IBM_model_1.likelihoods)
#plt.show()

# IBM_model_2 = IBM(model, corpus.training_english, corpus.training_french)
# IBM_model_2.load_test_sentences(corpus.testing_english, corpus.testing_french)
# IBM_model_2.em_algorithm(num_iteration)
# test_alignments = IBM_model_2.viterbi_alignment()
# IBM_model_2.calculate_aer(gold_standard_filepath, test_alignments)

# t, q = EM_algorithm(corpus.training_english[:1000], corpus.training_french[:1000])
