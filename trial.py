from __future__ import print_function
from pythtb import tb_model # import TB model class
import numpy as np
import matplotlib.pyplot as plt
#import io,os
#import sys
#import h5py
from tqdm import tqdm
from includeables import functions as f

f.plotGZ(kpoints=500,l1z=-0.001,l1=0.005,l3=0.)
plt.tight_layout()
f.plotGZ(kpoints=500,l1z=-0.001)