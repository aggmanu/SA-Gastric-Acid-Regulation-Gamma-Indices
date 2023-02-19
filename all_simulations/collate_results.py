import numpy as np
import os


grid_reso = 20
N_pars = 6

N_samples = 50000

all_outs = np.zeros((N_pars, grid_reso, N_samples))

f = open('swarm_failed.sh', 'w')
f.write('#!/bin/sh')

num_cores = 32


for p_idx in range(N_pars):

    for grid_idx in range(grid_reso):
        
        out_file = 'outs_'+str(p_idx)+'_'+str(grid_idx)+'.npy'
        #if not os.path.isfile(out_file):

        #    f.write('\n')
        #    f.write('python3 analyze_Owen.py '+str(p_idx)+' '+str(grid_idx)+' '+str(grid_reso)+' '+str(num_cores))
        #    print(p_idx, grid_idx, 'not found')
        #    continue

        outs = np.squeeze(np.load(out_file))

        par_file = 'pars_'+str(p_idx)+'_'+str(grid_idx)+'.npy'
        pars = np.load(par_file)


        all_outs[p_idx, grid_idx, :] = outs


np.save('all_conditional_outs.npy', all_outs)



