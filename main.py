from data_load import ParallelCorpus
from IBM import IBM

english_training_filepath = "./training/hansards.36.2.e"
french_training_filepath = "./training/hansards.36.2.f"
english_testing_filepath = "./validation/dev.e"
french_testing_filepath = "./validation/dev.f"
gold_standard_filepath = "./validation/dev.wa.nonullalign"
model = 1
num_iteration = 10

corpus = ParallelCorpus(english_training_filepath, french_training_filepath, \
                        english_testing_filepath, french_testing_filepath)

print("===========")
print("IBM model 1")
print("===========")
IBM_model_1 = IBM(model, corpus.training_english, corpus.training_french)
IBM_model_1.load_test_sentences(corpus.testing_english, corpus.testing_french)
IBM_model_1.em_algorithm(num_iteration)
print("==========================")
print("Starting Viterbi Alignment")
print("==========================")
test_alignments = IBM_model_1.viterbi_alignment()
print("===")
print("AER")
print("===")
IBM_model_1.calculate_aer(gold_standard_filepath, test_alignments)

print("===========")
print("IBM model 2")
print("===========")
IBM_model_2 = IBM(model, corpus.training_english, corpus.training_french)
IBM_model_2.load_test_sentences(corpus.testing_english, corpus.testing_french)
IBM_model_2.em_algorithm(num_iteration)
print("==========================")
print("Starting Viterbi Alignment")
print("==========================")
test_alignments = IBM_model_2.viterbi_alignment()
print("===")
print("AER")
print("===")
IBM_model_2.calculate_aer(gold_standard_filepath, test_alignments)
