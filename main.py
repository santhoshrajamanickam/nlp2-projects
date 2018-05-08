from data_load import ParallelCorpus
from IBM import IBM
import matplotlib.pyplot as plt

english_training_filepath = "./training/hansards.36.2.e"
french_training_filepath = "./training/hansards.36.2.f"
english_testing_filepath = "./validation/dev.e"
french_testing_filepath = "./validation/dev.f"
gold_standard_filepath = "./validation/dev.wa.nonullalign"
num_iteration = 10

corpus = ParallelCorpus(english_training_filepath, french_training_filepath, \
                        english_testing_filepath, french_testing_filepath)

IBM_model_1_EM = IBM('IBM1', corpus, gold_standard=gold_standard_filepath, limit=10)
IBM_model_1_EM.run_epoch(num_iteration,'EM')
plt.figure(1)
plt.plot(range(len(IBM_model_1_EM.log_likelihood)), IBM_model_1_EM.log_likelihood)
plt.savefig('./output/IBM_model_1_EM_likelihood.png', dpi=100)
plt.figure(2)
plt.plot(range(len(IBM_model_1_EM.aer)), IBM_model_1_EM.aer)
plt.savefig('./output/IBM_model_1_EM_aer.png', dpi=100)
log_likelihood_file = open('./output/log_likelihood.txt', 'w')
log_likelihood_file.write("IBM_model_1_EM: \n")
for item in IBM_model_1_EM.log_likelihood:
    log_likelihood_file.write("%s " % item)
log_likelihood_file.write("\n\n")
aer_file = open('./output/aer.txt', 'w')
aer_file.write("IBM_model_1_EM: \n")
for item in IBM_model_1_EM.aer:
    aer_file.write("%s " % item)
aer_file.write("\n\n")

IBM_model_1_Bayesian = IBM('IBM1', corpus, gold_standard=gold_standard_filepath, limit=10)
IBM_model_1_Bayesian.run_epoch(num_iteration,'VI', alpha=0.001)
plt.figure(3)
plt.plot(range(len(IBM_model_1_Bayesian.log_likelihood)), IBM_model_1_Bayesian.log_likelihood)
plt.savefig('./output/IBM_model_1_Bayesian_likelihood.png', dpi=100)
plt.figure(4)
plt.plot(range(len(IBM_model_1_Bayesian.aer)), IBM_model_1_Bayesian.aer)
plt.savefig('./output/IBM_model_1_Bayesian_aer.png', dpi=100)
plt.figure(5)
plt.plot(range(len(IBM_model_1_Bayesian.elbo)), IBM_model_1_Bayesian.elbo)
plt.savefig('./output/IBM_model_1_Bayesian_elbo.png', dpi=100)
log_likelihood_file.write("IBM_model_1_Bayesian: \n")
for item in IBM_model_1_Bayesian.log_likelihood:
    log_likelihood_file.write("%s " % item)
log_likelihood_file.write("\n\n")
aer_file.write("IBM_model_1_Bayesian: \n")
for item in IBM_model_1_Bayesian.aer:
    aer_file.write("%s " % item)
aer_file.write("\n\n")
elbo_file = open('./output/elbo.txt', 'w')
elbo_file.write("IBM_model_1_Bayesian: \n")
for item in IBM_model_1_Bayesian.elbo:
    elbo_file.write("%s " % item)
elbo_file.write("\n\n")

IBM_model_2_EM_uniform = IBM('IBM2', corpus, gold_standard=gold_standard_filepath, limit=10, initialization='uniform')
IBM_model_2_EM_uniform.run_epoch(num_iteration,'EM')
plt.figure(6)
plt.plot(range(len(IBM_model_2_EM_uniform.log_likelihood)), IBM_model_2_EM_uniform.log_likelihood)
plt.savefig('./output/IBM_model_2_EM_uniform_likelihood.png', dpi=100)
plt.figure(7)
plt.plot(range(len(IBM_model_2_EM_uniform.aer)), IBM_model_2_EM_uniform.aer)
plt.savefig('./output/IBM_model_2_EM_uniform_aer.png', dpi=100)
log_likelihood_file.write("IBM_model_2_EM_uniform: \n")
for item in IBM_model_2_EM_uniform.log_likelihood:
    log_likelihood_file.write("%s " % item)
log_likelihood_file.write("\n\n")
aer_file.write("IBM_model_2_EM_uniform: \n")
for item in IBM_model_2_EM_uniform.aer:
    aer_file.write("%s " % item)
aer_file.write("\n\n")

IBM_model_2_EM_random = IBM('IBM2', corpus, gold_standard=gold_standard_filepath, limit=10, initialization='random')
IBM_model_2_EM_random.run_epoch(num_iteration,'EM')
plt.figure(8)
plt.plot(range(len(IBM_model_2_EM_random.log_likelihood)), IBM_model_2_EM_random.log_likelihood)
plt.savefig('./output/IBM_model_2_EM_random_likelihood.png', dpi=100)
plt.figure(9)
plt.plot(range(len(IBM_model_2_EM_random.aer)), IBM_model_2_EM_random.aer)
plt.savefig('./output/IBM_model_2_EM_random_aer.png', dpi=100)
log_likelihood_file.write("IBM_model_2_EM_random: \n")
for item in IBM_model_2_EM_random.log_likelihood:
    log_likelihood_file.write("%s " % item)
log_likelihood_file.write("\n\n")
aer_file.write("IBM_model_2_EM_random: \n")
for item in IBM_model_2_EM_random.aer:
    aer_file.write("%s " % item)
aer_file.write("\n\n")

IBM_model_2_EM_pretrained = IBM('IBM2', corpus, gold_standard=gold_standard_filepath, limit=10, initialization='IBM1', pretrained_t=IBM_model_1_EM.t)
IBM_model_2_EM_pretrained.run_epoch(num_iteration,'EM')
plt.figure(10)
plt.plot(range(len(IBM_model_2_EM_pretrained.log_likelihood)), IBM_model_2_EM_pretrained.log_likelihood)
plt.savefig('./output/IBM_model_2_EM_pretrained_likelihood.png', dpi=100)
plt.figure(11)
plt.plot(range(len(IBM_model_2_EM_pretrained.aer)), IBM_model_2_EM_pretrained.aer)
plt.savefig('./output/IBM_model_2_EM_pretrained_aer.png', dpi=100)
log_likelihood_file.write("IBM_model_2_EM_pretrained: \n")
for item in IBM_model_2_EM_pretrained.log_likelihood:
    log_likelihood_file.write("%s " % item)
log_likelihood_file.write("\n\n")
aer_file.write("IBM_model_2_EM_pretrained: \n")
for item in IBM_model_2_EM_pretrained.aer:
    aer_file.write("%s " % item)
aer_file.write("\n\n")

log_likelihood_file.close()
aer_file.close()
elbo_file.close()


