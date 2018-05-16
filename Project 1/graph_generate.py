import matplotlib.pyplot as plt

log_likelihood = [-37713814.48205021 , -11626014.917077472, -8625235.209512083, -7716649.050278235, -7406649.249443135, -7268471.283150121, \
                  -7195723.752442893, -7153066.842293784, -7126067.542868832, -7107945.5987300975, -7095207.558974577, -7085909.200921279]
plt.figure(1)
plt.plot(range(len(log_likelihood)), log_likelihood, label='IBM_model_1_EM')
plt.legend()
plt.savefig('./output/likelihood.png', dpi=100)
log_likelihood = [-37713814.48205021, -14215232.637208374, -9968870.894455973, -9225359.80100888, -8966524.515333928, -8846330.334791394, \
                  -8781198.824984372, -8742140.209281279, -8716964.473581467, -8699861.285413876, -8687738.625336269, -8678841.124353]
plt.figure(1)
plt.plot(range(len(log_likelihood)), log_likelihood, label='IBM_model_1_Bayesian')
plt.legend()
plt.savefig('./output/likelihood.png', dpi=100)
log_likelihood = [-37713814.48205021, -24949312.74840014, -19319164.328306876, -17258735.565544076, -16647685.591037791, -16459578.061356155, \
                  -16381577.554533375, -16342045.865318934, -16319380.222348565, -16305227.00899818, -16295857.332590448, -16289384.691899713]
plt.figure(1)
plt.plot(range(len(log_likelihood)), log_likelihood, label='IBM_model_2_EM_uniform')
plt.legend()
plt.savefig('./output/likelihood.png', dpi=100)
log_likelihood = [-13057608.075035842, -25204944.708866686, -19766465.31179983, -17554421.74589033, -16797097.226134714, -16539744.209119897, \
                  -16431356.224762524, -16376067.615186956, -16343513.311556188, -16322788.719237925, -16309115.024876414, -16299658.109973347]
plt.figure(1)
plt.plot(range(len(log_likelihood)), log_likelihood, label='IBM_model_2_EM_random')
plt.legend()
plt.savefig('./output/likelihood.png', dpi=100)
log_likelihood = [-28293725.68461123, -18057329.887734096, -17131859.778173298, -16882709.98837662, -16751857.13395076, -16663423.373306891, \
                  -16597919.814050425, -16547424.844494171, -16508130.617362661, -16476854.184102349, -16451390.3119435, -16430440.039734283]
plt.figure(1)
plt.plot(range(len(log_likelihood)), log_likelihood, label='IBM_model_2_EM_pretrained')
plt.legend()
plt.title('Log-likelihood')
plt.savefig('./output/likelihood.png', dpi=100)

elbo = [1980137.444, 50961364.4977, 57730514.1775, 60192239.3076, 61350645.1103, 61948368.9247, 62276435.9733, 62465804.0881,
        62581477.6056, 62655064.6282, 62704679.6234, 62739753.3034]

plt.figure(2)
plt.plot(range(len(elbo)), elbo, label='IBM_model_1_Bayesian')
plt.title('ELBO')
plt.legend()
plt.savefig('./output/IBM_model_1_Bayesian_elbo.png', dpi=100)
