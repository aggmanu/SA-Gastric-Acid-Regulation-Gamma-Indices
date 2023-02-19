import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['text.usetex'] = True


all_outs = np.load('all_conditional_outs.npy')

labels = [\
	r'$H_L$',\
	r'$I_L$',\
	r'$k_{HI}$',\
	r'$k_{AB}$',\
	r'$\delta_{HI}$',\
	r'$\delta_{AB}$',\
	]

n_pars, n_grid, n_samples = all_outs.shape



cond_mean = np.zeros((n_pars, n_grid))
cond_var = np.zeros((n_pars, n_grid))
cond_skew = np.zeros((n_pars, n_grid))
cond_kurt = np.zeros((n_pars, n_grid))

fig, axs = plt.subplots(2,2)

grid_plot = list(range(n_grid))

for p_idx in range(n_pars):
    
    for grid in range(n_grid):
        
        this_violin = all_outs[p_idx, grid, :]
        cond_mean[p_idx, grid] = np.mean(this_violin)
        cond_var[p_idx, grid] = np.var(this_violin)
        cond_skew[p_idx, grid] = stats.skew(this_violin)
        cond_kurt[p_idx, grid] = stats.kurtosis(this_violin)
        
        
    
    axs[0,0].plot(grid_plot, cond_mean[p_idx,:])
    axs[0,1].plot(grid_plot, cond_var[p_idx,:])
    axs[1,0].plot(grid_plot, cond_skew[p_idx,:])
    axs[1,1].plot(grid_plot, cond_kurt[p_idx,:], label=labels[p_idx])

axs[0,0].set_ylabel('cond. expected value', fontsize=14)
axs[0,1].set_ylabel('cond. variance', fontsize=14)
axs[1,0].set_ylabel('cond. skewness', fontsize=14)
axs[1,1].set_ylabel('cond. kurtosis', fontsize=14)

axs[0,0].set_xticks([0, n_grid-1])
axs[0,1].set_xticks([0, n_grid-1])
axs[1,0].set_xticks([0, n_grid-1])
axs[1,1].set_xticks([0, n_grid-1])

axs[0,0].set_xticklabels(['min par', 'max par'])
axs[0,1].set_xticklabels(['min par', 'max par'])
axs[1,0].set_xticklabels(['min par', 'max par'])
axs[1,1].set_xticklabels(['min par', 'max par'])

axs[1,1].legend()

plt.tight_layout()


plt.savefig('plot_conditionals.pdf', format='pdf')

plt.show()





