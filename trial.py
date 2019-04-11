from __future__ import print_function
from pythtb import tb_model # import TB model class
import numpy as np
import matplotlib.pyplot as plt
import io,os
import sys
import h5py
from tqdm import tqdm
from includeables import functions as f

FST_bulk=f.setup_model3d()
(k_vec,k_dist,k_node) = f.symGMcut3d(FST_bulk,0.0657793/2.,0.03,81)
(evals,evecs)=FST_bulk.solve_all(k_vec,eig_vectors=True)
f.plotbulkbands(evals,evecs,k_vec,k_dist,k_node,[-0.050,0.005])