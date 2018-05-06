from data_load import ParallelCorpus
from IBM import IBM

english_training_filepath = "./training/hansards.36.2.e"
french_training_filepath = "./training/hansards.36.2.f"
english_testing_filepath = "./validation/dev.e"
french_testing_filepath = "./validation/dev.f"
gold_standard_filepath = "./validation/dev.wa.nonullalign"
num_iteration = 10

corpus = ParallelCorpus(english_training_filepath, french_training_filepath, \
                        english_testing_filepath, french_testing_filepath)

IBM_model_1 = IBM('IBM1', corpus, limit=1000)
IBM_model_1.run_epoch(num_iteration,'EM')
test_alignments = IBM_model_1.viterbi_alignment()
IBM_model_1.calculate_aer(gold_standard_filepath, test_alignments)
