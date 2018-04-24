from data_load import ParallelCorpus
from EM_algorithm import EM_algorithm

english_training_filepath = "./training/hansards.36.2.e"
french_training_filepath = "./training/hansards.36.2.f"

corpus = ParallelCorpus(english_training_filepath, french_training_filepath)
t, q = EM_algorithm(corpus.english_sentences[:100], corpus.french_sentences[:100])
