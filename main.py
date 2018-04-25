from data_load import ParallelCorpus
from IBM import IBM

english_training_filepath = "./training/hansards.36.2.e"
french_training_filepath = "./training/hansards.36.2.f"
english_testing_filepath = "./validation/dev.e"
french_testing_filepath = "./validation/dev.f"
model = 2
num_iteration = 10

corpus = ParallelCorpus(english_training_filepath, french_training_filepath, \
                        english_testing_filepath, french_testing_filepath)

IBM_model_2 = IBM(model, corpus.training_english[:100], corpus.training_french[:100])
IBM_model_2.load_test_sentences(corpus.testing_english, corpus.testing_french)
IBM_model_2.em_algorithm(num_iteration)
IBM_model_2.viterbi_alignment()

# t, q = EM_algorithm(corpus.training_english[:1000], corpus.training_french[:1000])
