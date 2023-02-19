
f = open('swarm_compute.sh', 'w')

f.write('#!/bin/sh')


grid_reso = 20
num_cores = 12

N_pars = 6

for p_idx in range(N_pars):

    for grid_idx in range(grid_reso):
    
        f.write('\n')
        f.write('python3 analyze_Owen.py '+str(p_idx)+' '+str(grid_idx)+' '+str(grid_reso)+' '+str(num_cores))
    
f.close()


