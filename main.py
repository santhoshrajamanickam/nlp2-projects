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


print("==============")
print("IBM_model_1_EM")
print("==============")
IBM_model_1_EM = IBM('IBM1', corpus, gold_standard=gold_standard_filepath)
IBM_model_1_EM.run_epoch(num_iteration,'EM')
final_alignment = IBM_model_1_EM.viterbi_alignment()
IBM_model_1_EM.write_naacl_format(final_alignment,'./output/ibm1.mle.naacl')
plt.figure(1)
plt.plot(range(len(IBM_model_1_EM.log_likelihood)), IBM_model_1_EM.log_likelihood, label='IBM_model_1_EM')
plt.savefig('./output/likelihood.png', dpi=100)
plt.figure(2)
plt.plot(range(len(IBM_model_1_EM.aer)), IBM_model_1_EM.aer, label='IBM_model_1_EM')
plt.savefig('./output/aer.png', dpi=100)
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


print("====================")
print("IBM_model_1_Bayesian")
print("====================")
IBM_model_1_Bayesian = IBM('IBM1', corpus, gold_standard=gold_standard_filepath)
IBM_model_1_Bayesian.run_epoch(num_iteration,'VI', alpha=0.001)
final_alignment = IBM_model_1_Bayesian.viterbi_alignment()
IBM_model_1_Bayesian.write_naacl_format(final_alignment,'./output/ibm1.vb.naacl')
plt.figure(1)
plt.plot(range(len(IBM_model_1_Bayesian.log_likelihood)), IBM_model_1_Bayesian.log_likelihood, label='IBM_model_1_Bayesian')
plt.savefig('./output/likelihood.png', dpi=100)
plt.figure(2)
plt.plot(range(len(IBM_model_1_Bayesian.aer)), IBM_model_1_Bayesian.aer, label='IBM_model_1_Bayesian')
plt.savefig('./output/aer.png', dpi=100)
plt.figure(3)
plt.plot(range(len(IBM_model_1_Bayesian.elbo)), IBM_model_1_Bayesian.elbo, label='IBM_model_1_Bayesian')
plt.title('ELBO')
plt.legend()
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


print("======================")
print("IBM_model_2_EM_uniform")
print("======================")
IBM_model_2_EM_uniform = IBM('IBM2', corpus, gold_standard=gold_standard_filepath, initialization='uniform')
IBM_model_2_EM_uniform.run_epoch(num_iteration,'EM')
final_alignment = IBM_model_2_EM_uniform.viterbi_alignment()
IBM_model_2_EM_uniform.write_naacl_format(final_alignment,'./output/ibm2.mle.uniform.naacl')
plt.figure(1)
plt.plot(range(len(IBM_model_2_EM_uniform.log_likelihood)), IBM_model_2_EM_uniform.log_likelihood, label='IBM_model_2_EM_uniform')
plt.savefig('./output/likelihood.png', dpi=100)
plt.figure(2)
plt.plot(range(len(IBM_model_2_EM_uniform.aer)), IBM_model_2_EM_uniform.aer, label='IBM_model_2_EM_uniform')
plt.savefig('./output/aer.png', dpi=100)
log_likelihood_file.write("IBM_model_2_EM_uniform: \n")
for item in IBM_model_2_EM_uniform.log_likelihood:
    log_likelihood_file.write("%s " % item)
log_likelihood_file.write("\n\n")
aer_file.write("IBM_model_2_EM_uniform: \n")
for item in IBM_model_2_EM_uniform.aer:
    aer_file.write("%s " % item)
aer_file.write("\n\n")

print("=====================")
print("IBM_model_2_EM_random")
print("=====================")
IBM_model_2_EM_random = IBM('IBM2', corpus, gold_standard=gold_standard_filepath, initialization='random')
IBM_model_2_EM_random.run_epoch(num_iteration,'EM')
final_alignment = IBM_model_2_EM_random.viterbi_alignment()
IBM_model_2_EM_random.write_naacl_format(final_alignment,'./output/ibm2.mle.random.naacl')
plt.figure(1)
plt.plot(range(len(IBM_model_2_EM_random.log_likelihood)), IBM_model_2_EM_random.log_likelihood, label='IBM_model_2_EM_random')
plt.savefig('./output/likelihood.png', dpi=100)
plt.figure(2)
plt.plot(range(len(IBM_model_2_EM_random.aer)), IBM_model_2_EM_random.aer, label='IBM_model_2_EM_random')
plt.savefig('./output/aer.png', dpi=100)
log_likelihood_file.write("IBM_model_2_EM_random: \n")
for item in IBM_model_2_EM_random.log_likelihood:
    log_likelihood_file.write("%s " % item)
log_likelihood_file.write("\n\n")
aer_file.write("IBM_model_2_EM_random: \n")
for item in IBM_model_2_EM_random.aer:
    aer_file.write("%s " % item)
aer_file.write("\n\n")

print("=========================")
print("IBM_model_2_EM_pretrained")
print("=========================")
IBM_model_2_EM_pretrained = IBM('IBM2', corpus, gold_standard=gold_standard_filepath, initialization='IBM1', pretrained_t=IBM_model_1_EM.t)
IBM_model_2_EM_pretrained.run_epoch(num_iteration,'EM')
final_alignment = IBM_model_2_EM_pretrained.viterbi_alignment()
IBM_model_2_EM_pretrained.write_naacl_format(final_alignment,'./output/ibm2.mle.pretrained.naacl')
plt.figure(1)
plt.plot(range(len(IBM_model_2_EM_pretrained.log_likelihood)), IBM_model_2_EM_pretrained.log_likelihood, label='IBM_model_2_EM_pretrained')
plt.legend()
plt.title('Log Likelihood')
plt.savefig('./output/likelihood.png', dpi=100)
plt.figure(2)
plt.plot(range(len(IBM_model_2_EM_pretrained.aer)), IBM_model_2_EM_pretrained.aer, label='IBM_model_2_EM_pretrained')
plt.title('AER')
plt.legend()
plt.savefig('./output/aer.png', dpi=100)
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


