from __future__ import print_function
from pythtb import tb_model # import TB model class
import numpy as np
import matplotlib.pyplot as plt
import io,os
import sys
import h5py
from tqdm import tqdm

def show():
	plt.show(block=False)

def setup_model3dasbefore(scalefactorC=1.,scalefactorX=1.,scalefactorhopp=1.,scalefactorhopd=1.,scalefactorgap=1.,scalefactordxytop=1.,scalefactorB=1.):
    # define lattice vectors
    lat=[[1.0,0.0,0.0],[0.0,0.1,0.0],[0.0,0.0,0.1]]
    # define coordinates of orbitals
    orb=[[0.5,0.5,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
    # make three dimensional tight-binding model
    my_model=tb_model(3,3,lat,orb,[0,1,2],2)
    # set model parameters
    M01 = 0.04*1.
    M11 = 2.4*1.
    M21 = -0.8*scalefactorhopp
    M02 = 0.
    M12 = -3.8*1.
    M22 = 0.14*scalefactorhopd
    M03 = -0.002*-4.*scalefactordxytop
    M13 = -0.2*1.
    M23 = 0.002*1. #looks unchanged
    A = -4.4*1.
    B = 2.2*1.*scalefactorB
    C = 0.004*1.*scalefactorC
    X = 0.412*1.*scalefactorX
    L1 = 0.0105*1.
    L2 = 0.003*1.
    L3 = 0.019*scalefactorgap
    #onsite
    my_model.set_onsite([M01+4.0*M11+2.0*M21,M02+4.0*M12+2.0*M22,M02+4.0*M12+2.0*M22,M03+4.0*M13+2.0*M23], mode ="reset")
    #my_model.set_onsite([0.,0.,0.,0.])
    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    #pz to pz
    my_model.set_hop(-M11, 0, 0, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M11, 0, 0, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M21, 0, 0, [ 0, 0, 1], mode ="reset")
    #dyz to dyz
    my_model.set_hop(-M12-B, 1, 1, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M12+B, 1, 1, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M22, 1, 1, [ 0, 0, 1], mode ="reset")
    #dxz to dxz
    my_model.set_hop(-M12+B, 2, 2, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M12-B, 2, 2, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M22, 2, 2, [ 0, 0, 1], mode ="reset")
    #dxy to dxy
    my_model.set_hop(-M13, 3, 3, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M13, 3, 3, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M23, 3, 3, [ 0, 0, 1], mode ="reset")
    #pz to dyz
    #n.b. to check the sign of the following term : hopping integral from orbital 1 at lattice [0,1,0] to orbital 0 at origin, and  -ie^ikz +hc ~ 2sinkz
    my_model.set_hop(X/2.0, 0, 1, [ 0, -1, 0], mode ="reset")
    my_model.set_hop(-X/2.0, 0, 1, [ 0, 1, 0], mode ="reset")
    #pz to dxz
    my_model.set_hop(X/2.0, 0, 2, [ -1, 0, 0], mode ="reset")
    my_model.set_hop(-X/2.0, 0, 2, [ 1, 0, 0], mode ="reset")
    #dyz to dxz
    my_model.set_hop(-A/4.0, 1, 2, [ 1, 1, 0], mode ="reset")
    my_model.set_hop(-A/4.0, 1, 2, [-1,-1, 0], mode ="reset")
    my_model.set_hop( A/4.0, 1, 2, [ 1,-1, 0], mode ="reset")
    my_model.set_hop( A/4.0, 1, 2, [-1, 1, 0], mode ="reset")
    ##check these two terms
    #dyz to dxy
    my_model.set_hop(-C/4.0, 1, 3, [ 1, 0, 1], mode ="reset")
    my_model.set_hop(-C/4.0, 1, 3, [-1, 0,-1], mode ="reset")
    my_model.set_hop( C/4.0, 1, 3, [ 1, 0,-1], mode ="reset")
    my_model.set_hop( C/4.0, 1, 3, [-1, 0, 1], mode ="reset")
    #dxz to dxy
    my_model.set_hop(-C/4.0, 2, 3, [ 0, 1, 1], mode ="reset")
    my_model.set_hop(-C/4.0, 2, 3, [ 0,-1,-1], mode ="reset")
    my_model.set_hop( C/4.0, 2, 3, [ 0, 1,-1], mode ="reset")
    my_model.set_hop( C/4.0, 2, 3, [ 0,-1, 1], mode ="reset")
    # set SOC
 
    my_model.set_hop([0.,0.,0., L1*1.0j], 1, 2, [0.,0.,0.], mode="add")
    my_model.set_hop([0.,0.,-L2*1.0j,0.], 1, 3, [0.,0.,0.], mode="add")
    my_model.set_hop([0., L2*1.0j,0.,0.], 2, 3, [0.,0.,0.], mode="add")
 
    my_model.set_hop([0., L3*1.0j/2.0,0.,0.], 0, 1, [0.,0., 1.], mode="add")
    my_model.set_hop([0.,-L3*1.0j/2.0,0.,0.], 0, 1, [0.,0.,-1.], mode="add")
    my_model.set_hop([0.,0.,-L3*1.0j/2.0,0.], 0, 2, [0.,0., 1.], mode="add")
    my_model.set_hop([0.,0., L3*1.0j/2.0,0.], 0, 2, [0.,0.,-1.], mode="add")
 
    return my_model

def setup_model3dwithevenL3(scalefactor,scalefactorhopp=1.,scalefactorhopd=1.,scalefactorgap=-5.,scalefactordxytop=1.):
    # define lattice vectors
    lat=[[1.0,0.0,0.0],[0.0,0.1,0.0],[0.0,0.0,0.1]]
    # define coordinates of orbitals
    orb=[[0.5,0.5,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
    # make three dimensional tight-binding model
    my_model=tb_model(3,3,lat,orb,[0,1,2],2)
    # set model parameters
    M01 = 0.04*1.*scalefactor
    M11 = 2.4*1.
    M21 = -0.8*scalefactorhopp
    M02 = 0.
    M12 = -3.8*1.
    M22 = 0.14**scalefactorhopd
    M03 = -0.002*-4.*scalefactordxytop
    M13 = -0.2*1.
    M23 = 0.002*1.
    A = -4.4*1.
    B = 2.2*1.
    C = 0.004*1.
    X = 0.412*1.
    L1 = 0.0105*1.
    L2 = 0.003*1.
    L3 = 0.019*scalefactorgap
    #onsite
    my_model.set_onsite([M01+4.0*M11+2.0*M21,M02+4.0*M12+2.0*M22,M02+4.0*M12+2.0*M22,M03+4.0*M13+2.0*M23], mode ="reset")
    #my_model.set_onsite([0.,0.,0.,0.])
    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    #pz to pz
    my_model.set_hop(-M11, 0, 0, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M11, 0, 0, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M21, 0, 0, [ 0, 0, 1], mode ="reset")
    #dyz to dyz
    my_model.set_hop(-M12-B, 1, 1, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M12+B, 1, 1, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M22, 1, 1, [ 0, 0, 1], mode ="reset")
    #dxz to dxz
    my_model.set_hop(-M12+B, 2, 2, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M12-B, 2, 2, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M22, 2, 2, [ 0, 0, 1], mode ="reset")
    #dxy to dxy
    my_model.set_hop(-M13, 3, 3, [ 1, 0, 0], mode ="reset")
    my_model.set_hop(-M13, 3, 3, [ 0, 1, 0], mode ="reset")
    my_model.set_hop(-M23, 3, 3, [ 0, 0, 1], mode ="reset")
    #pz to dyz
    #n.b. to check the sign of the following term : hopping integral from orbital 1 at lattice [0,1,0] to orbital 0 at origin, and  -ie^ikz +hc ~ 2sinkz
    my_model.set_hop(X/2.0, 0, 1, [ 0, -1, 0], mode ="reset")
    my_model.set_hop(-X/2.0, 0, 1, [ 0, 1, 0], mode ="reset")
    #pz to dxz
    my_model.set_hop(X/2.0, 0, 2, [ -1, 0, 0], mode ="reset")
    my_model.set_hop(-X/2.0, 0, 2, [ 1, 0, 0], mode ="reset")
    #dyz to dxz
    my_model.set_hop(-A/4.0, 1, 2, [ 1, 1, 0], mode ="reset")
    my_model.set_hop(-A/4.0, 1, 2, [-1,-1, 0], mode ="reset")
    my_model.set_hop( A/4.0, 1, 2, [ 1,-1, 0], mode ="reset")
    my_model.set_hop( A/4.0, 1, 2, [-1, 1, 0], mode ="reset")
    ##check these two terms
    #dyz to dxy
    my_model.set_hop(-C/4.0, 1, 3, [ 1, 0, 1], mode ="reset")
    my_model.set_hop(-C/4.0, 1, 3, [-1, 0,-1], mode ="reset")
    my_model.set_hop( C/4.0, 1, 3, [ 1, 0,-1], mode ="reset")
    my_model.set_hop( C/4.0, 1, 3, [-1, 0, 1], mode ="reset")
    #dxz to dxy
    my_model.set_hop(-C/4.0, 2, 3, [ 0, 1, 1], mode ="reset")
    my_model.set_hop(-C/4.0, 2, 3, [ 0,-1,-1], mode ="reset")
    my_model.set_hop( C/4.0, 2, 3, [ 0, 1,-1], mode ="reset")
    my_model.set_hop( C/4.0, 2, 3, [ 0,-1, 1], mode ="reset")
    # set SOC
 
    my_model.set_hop([0.,0.,0., L1*1.0j], 1, 2, [0.,0.,0.], mode="add")
    my_model.set_hop([0.,0.,-L2*1.0j,0.], 1, 3, [0.,0.,0.], mode="add")
    my_model.set_hop([0., L2*1.0j,0.,0.], 2, 3, [0.,0.,0.], mode="add")
 
    my_model.set_hop([0., L3*1.0/2.0,0.,0.], 0, 1, [0.,0., 1.], mode="add")
    my_model.set_hop([0., L3*1.0/2.0,0.,0.], 0, 1, [0.,0.,-1.], mode="add")
    my_model.set_hop([0.,0., L3*1.0/2.0,0.], 0, 2, [0.,0., 1.], mode="add")
    my_model.set_hop([0.,0., L3*1.0/2.0,0.], 0, 2, [0.,0.,-1.], mode="add")
 
    return my_model

def setup_model3d(M02=-0.012,M12=-3.8,M22=0.00425,M01=0.002,M11=2.4,M21=-0.006,M03=-0.002,M13=-0.2,M23=0.0001,A = -4.4*1.,B = 2.2*1.,C = 0.004*1.,X = 0.412*1.,L1=0.01,L2=0.001,L3=0.004):
	# define lattice vectors
	lat=[[1.0,0.0,0.0],[0.0,0.1,0.0],[0.0,0.0,0.1]]
	# define coordinates of orbitals
	orb=[[0.5,0.5,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
	# make three dimensional tight-binding model
	my_model=tb_model(3,3,lat,orb,[0,1,2],2)
	# set model parameters
	# M01 = 0.04*1.
	# M11 = 2.4*1.
	# M21 = -0.2*1.
	# M02 = 0.
	# M12 = -3.8*1.
	# M22 = 0.14*1.
	# M03 = -0.002*-4.
	# M13 = -0.2*1.
	# M23 = 0.002*1.
	# A = -4.4*1.
	# B = 2.2*1.
	# C = 0.004*1.
	# X = 0.412*1.
	# L1 = 0.0105*1.0*scalefactor1
	# L2 = 0.003*1.0*scalefactor2
	# L3 = 0.019*1.0*scalefactor3
	# Lsymbr = 0.000001
	#LR = 0.005*0.
	#onsite
	my_model.set_onsite([M01+4.0*M11+2.0*M21,M02+4.0*M12+2.0*M22,M02+4.0*M12+2.0*M22,M03+4.0*M13+2.0*M23], mode ="reset")
	#my_model.set_onsite([0.,0.,0.,0.])
	# set hoppings (one for each connected pair of orbitals)
	# (amplitude, i, j, [lattice vector to cell containing j])
	#pz to pz
	my_model.set_hop(-M11, 0, 0, [ 1, 0, 0], mode ="reset")
	my_model.set_hop(-M11, 0, 0, [ 0, 1, 0], mode ="reset")
	my_model.set_hop(-M21, 0, 0, [ 0, 0, 1], mode ="reset")
	#dyz to dyz
	my_model.set_hop(-M12-B, 1, 1, [ 1, 0, 0], mode ="reset")
	my_model.set_hop(-M12+B, 1, 1, [ 0, 1, 0], mode ="reset")
	my_model.set_hop(-M22, 1, 1, [ 0, 0, 1], mode ="reset")
	#dxz to dxz
	my_model.set_hop(-M12+B, 2, 2, [ 1, 0, 0], mode ="reset")
	my_model.set_hop(-M12-B, 2, 2, [ 0, 1, 0], mode ="reset")
	my_model.set_hop(-M22, 2, 2, [ 0, 0, 1], mode ="reset")
	#dxy to dxy
	my_model.set_hop(-M13, 3, 3, [ 1, 0, 0], mode ="reset")
	my_model.set_hop(-M13, 3, 3, [ 0, 1, 0], mode ="reset")
	my_model.set_hop(-M23, 3, 3, [ 0, 0, 1], mode ="reset")
	#pz to dyz
	#n.b. to check the sign of the following term : hopping integral from orbital 1 at lattice [0,1,0] to orbital 0 at origin, and  -ie^ikz +hc ~ 2sinkz
	my_model.set_hop(X/2.0, 0, 1, [ 0, -1, 0], mode ="reset")
	my_model.set_hop(-X/2.0, 0, 1, [ 0, 1, 0], mode ="reset")
	#pz to dxz
	my_model.set_hop(X/2.0, 0, 2, [ -1, 0, 0], mode ="reset")
	my_model.set_hop(-X/2.0, 0, 2, [ 1, 0, 0], mode ="reset")
	#dyz to dxz
	my_model.set_hop(-A/4.0, 1, 2, [ 1, 1, 0], mode ="reset")
	my_model.set_hop(-A/4.0, 1, 2, [-1,-1, 0], mode ="reset")
	my_model.set_hop( A/4.0, 1, 2, [ 1,-1, 0], mode ="reset")
	my_model.set_hop( A/4.0, 1, 2, [-1, 1, 0], mode ="reset")
	##check these two terms
	#dyz to dxy
	my_model.set_hop(-C/4.0, 1, 3, [ 1, 0, 1], mode ="reset")
	my_model.set_hop(-C/4.0, 1, 3, [-1, 0,-1], mode ="reset")
	my_model.set_hop( C/4.0, 1, 3, [ 1, 0,-1], mode ="reset")
	my_model.set_hop( C/4.0, 1, 3, [-1, 0, 1], mode ="reset")
	#dxz to dxy
	my_model.set_hop(-C/4.0, 2, 3, [ 0, 1, 1], mode ="reset")
	my_model.set_hop(-C/4.0, 2, 3, [ 0,-1,-1], mode ="reset")
	my_model.set_hop( C/4.0, 2, 3, [ 0, 1,-1], mode ="reset")
	my_model.set_hop( C/4.0, 2, 3, [ 0,-1, 1], mode ="reset")
	# set SOC

	my_model.set_hop([0.,0.,0., L1*1.0j], 1, 2, [0.,0.,0.], mode="add")
	my_model.set_hop([0.,0.,-L2*1.0j,0.], 1, 3, [0.,0.,0.], mode="add")
	my_model.set_hop([0., L2*1.0j,0.,0.], 2, 3, [0.,0.,0.], mode="add")

	my_model.set_hop([0., L3*1.0j/2.0,0.,0.], 0, 1, [0.,0., 1.], mode="add")
	my_model.set_hop([0.,-L3*1.0j/2.0,0.,0.], 0, 1, [0.,0.,-1.], mode="add")
	my_model.set_hop([0.,0.,-L3*1.0j/2.0,0.], 0, 2, [0.,0., 1.], mode="add")
	my_model.set_hop([0.,0., L3*1.0j/2.0,0.], 0, 2, [0.,0.,-1.], mode="add")

	#set Rashba ... only implemented for x-hops now
	# my_model.set_hop([0.,0.,LR*1.j,0.], 0, 0, [1.,0.,0.] , mode="add")
	# my_model.set_hop([0.,0.,LR*1.j,0.], 1, 1, [1.,0.,0.] , mode="add")
	# my_model.set_hop([0.,0.,LR*1.j,0.], 2, 2, [1.,0.,0.] , mode="add")
	# my_model.set_hop([0.,0.,LR*1.j,0.], 3, 3, [1.,0.,0.] , mode="add")
	#small explicit sigma_y breaking term to separate 
	# my_model.set_onsite([0.,0.,0.,-Lsymbr], 0, mode="add")
	# my_model.set_onsite([0.,0.,0.,-Lsymbr], 1, mode="add")
	# my_model.set_onsite([0.,0.,0.,-Lsymbr], 2, mode="add")
	# my_model.set_onsite([0.,0.,0.,-Lsymbr], 3, mode="add")
	# my_model.set_onsite([0.,0.,Lsymbr,0.], 0, mode="add")
	# my_model.set_onsite([0.,0.,Lsymbr,0.], 1, mode="add")
	# my_model.set_onsite([0.,0.,Lsymbr,0.], 2, mode="add")
	# my_model.set_onsite([0.,0.,Lsymbr,0.], 3, mode="add")
	return my_model

def symGXcut3d(model,kz=0.,kmax=0.005,klength=21):
	return model.k_path([[-kmax,-kmax,kz],[0.,0.,kz],[kmax,kmax,kz]],klength,report=False)

def symGMcut3d(model,kz=0.,kmax=0.02,klength=21):
	return model.k_path([[-kmax,0.,kz],[0.,0.,kz],[kmax,0.,kz]],klength,report=False)

def setup_model2d(my_model,zlength=40):
	#make a 2D model by defining a super-cell of 100 orbitals in z direction => 100 layers
	FST_slab=my_model.cut_piece(zlength,2,glue_edgs=False)
	return FST_slab

def symGXcut2d(FST_slab,kmax=0.005,klength=21):
	return FST_slab.k_path([[-kmax,-kmax],[0.,0.],[kmax,kmax]],klength,report=False)

def symGMcut2d(FST_slab,kmax=0.02,klength=21):
	return FST_slab.k_path([[-kmax,0.],[0.,0.],[kmax,0.]],klength,report=False)

def plotbulkbands(evals,evecs,k_vec,k_dist,k_node,yrange):
	label=(r' ',r'$\Gamma$',r'$\rightarrow$ M')
	plt.figure(figsize=(8,3))
	plt.subplot(1,4,1)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node, label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$p_Z$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], marker= 'o', s = np.sum(np.abs(evecs[i])**2,2)[::,0]*20.)
		plt.plot(k_dist,evals[i],linewidth=0.1)
	plt.subplot(1,4,2)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{YZ}$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], marker= 'o', s = np.sum(np.abs(evecs[i])**2,2)[::,1]*20.)
		plt.plot(k_dist,evals[i],linewidth=0.1)
	plt.subplot(1,4,3)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{XZ}$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], marker= 'o', s = np.sum(np.abs(evecs[i])**2,2)[::,2]*20.)
		plt.plot(k_dist,evals[i],linewidth=0.1)
	plt.subplot(1,4,4)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{XY}$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], marker= 'o', s	= np.sum(np.abs(evecs[i])**2,2)[::,3]*20.)
		plt.plot(k_dist,evals[i],linewidth=0.1)
	plt.tight_layout()
	plt.show(block=False)

def plotslabbands(evals,evecs,k_vec,k_dist,k_node,yrange):
	label=(r' ',r'$\Gamma$',r'$\rightarrow M$')
	plt.figure(figsize=(10,5))
	plt.subplot(1,4,1)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node, label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$p_Z$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], color='k', marker='o', s = np.sum(np.abs(evecs[i,::,0::4])**2,(1,2))*20.)
		plt.plot(k_dist,evals[i],linewidth=0.5,color='k')
	plt.subplot(1,4,2)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{YZ}$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], color='k', marker='o', s = np.sum(np.abs(evecs[i,::,1::4])**2,(1,2))*20.)
		plt.plot(k_dist,evals[i],linewidth=0.5,color='k')
	plt.subplot(1,4,3)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{XZ}$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], color='k', marker='o', s = np.sum(np.abs(evecs[i,::,2::4])**2,(1,2))*20.)
		plt.plot(k_dist,evals[i],linewidth=0.5,color='k')
	plt.subplot(1,4,4)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{XY}$ component")
	plt.axhline(y=0,linewidth=0.5, color='k')
	for i in range(len(evals)):
		plt.scatter(k_dist,evals[i], color='k', marker='o', s	= np.sum(np.abs(evecs[i,::,3::4])**2,(1,2))*20.)
		plt.plot(k_dist,evals[i],linewidth=1.0,color='k')
	plt.tight_layout()
	plt.show(block=False)

def plotslabSurfaceStateOrbitalComponents(evals,evecs,k_vec,k_dist,k_node,eigindexrange,decaylength):
	label=(r' ',r'$\Gamma$',r'$\rightarrow M$')
	zlength=len(evecs[1,1])/4
	# decaylength=1.0
	decayfunc=np.exp(-np.arange(zlength)/decaylength)
	decayfunc=decayfunc/np.full(len(decayfunc),np.sqrt(np.sum(decayfunc)))
	eigcount=len(eigindexrange)
	count=0
	plt.figure(figsize=(10,7))
	for en in eigindexrange:
		plt.subplot(eigcount,4,4*count+1)
		plt.xlim([k_node[0],k_node[-1]])
		plt.xticks(k_node, label)
		plt.ylim(0.,0.01)
		for n in range(len(k_node)):
			plt.axvline(x=k_node[n],linewidth=0.5, color='k')
		plt.title(r"$p_z$ component for band #"+str(en))
		# plt.axhline(y=0,linewidth=0.5, color='k')
		# plt.axhline(y=1,linewidth=0.5, color='k')
		plt.plot(k_dist, np.sum(np.sum(np.abs(evecs[en,::,0::4])**2,2)*decayfunc,1),linewidth=1.5,color='k',marker='o')
		plt.subplot(eigcount,4,4*count+2)
		plt.xlim([k_node[0],k_node[-1]])
		plt.xticks(k_node, label)
		plt.ylim(0.,0.01)
		for n in range(len(k_node)):
			plt.axvline(x=k_node[n],linewidth=0.5, color='k')
		plt.title(r"$d_{yz}$ component for band #"+str(en))
		# plt.axhline(y=0,linewidth=0.5, color='k')
		# plt.axhline(y=1,linewidth=0.5, color='k')
		plt.scatter(k_dist, np.sum(np.sum(np.abs(evecs[en,::,1::4])**2,2)*decayfunc,1),linewidth=1.5,color='k',marker='o')
		plt.subplot(eigcount,4,4*count+3)
		plt.xlim([k_node[0],k_node[-1]])
		plt.xticks(k_node, label)
		plt.ylim(0.,0.01)
		for n in range(len(k_node)):
			plt.axvline(x=k_node[n],linewidth=0.5, color='k')
		plt.title(r"$d_{xz}$ component for band #"+str(en))
		# plt.axhline(y=0,linewidth=0.5, color='k')
		# plt.axhline(y=1,linewidth=0.5, color='k')
		plt.scatter(k_dist, np.sum(np.sum(np.abs(evecs[en,::,2::4])**2,2)*decayfunc,1),linewidth=1.5,color='k',marker='o')
		plt.subplot(eigcount,4,4*count+4)
		plt.xlim([k_node[0],k_node[-1]])
		plt.xticks(k_node, label)
		plt.ylim(0.,0.01)
		for n in range(len(k_node)):
			plt.axvline(x=k_node[n],linewidth=0.5, color='k')
		plt.title(r"$d_{xy}$ component for band #"+str(en))
		# plt.axhline(y=0,linewidth=0.5, color='k')
		# plt.axhline(y=1,linewidth=0.5, color='k')
		plt.scatter(k_dist, np.sum(np.sum(np.abs(evecs[en,::,3::4])**2,2)*decayfunc,1),linewidth=1.5,color='k',marker='o')
		count+=1
	plt.tight_layout()
	plt.show(block=False)

def plotsurfacebands(evals,evecs,k_vec,k_dist,k_node,yrange,decaylength=1.0):
	label=(r' ',r'$\Gamma$',r'$\rightarrow M$')
	zlength=len(evecs[1,1])/4
	decayfunc=np.exp(-np.arange(zlength)/decaylength)
	decayfunc=decayfunc/np.sqrt(np.sum(decayfunc))*20.
	plt.figure(figsize=(10,5))
	plt.subplot(1,4,1)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node, label)
	plt.axhline(y=0,linewidth=0.5, color='k')
	flagarr=np.empty(len(evals))
	for i in range(len(evals)):
		flagarr[i]=(max(evals[i])>yrange[0] and min(evals[i])<yrange[1])
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$p_z$ component")
	for i in range(len(evals)):
		if flagarr[i]:
			trial0=np.sum(np.abs(evecs[i,::,0::4])**2,2)
			plt.scatter(k_dist,evals[i], color='k', marker='o', s = np.einsum('ij,j',trial0,decayfunc))
			plt.plot(k_dist,evals[i],linewidth=0.05)
	plt.subplot(1,4,2)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	plt.axhline(y=0,linewidth=0.5, color='k')
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{yz}$ component")
	for i in range(len(evals)):
		if flagarr[i]:
			trial1=np.sum(np.abs(evecs[i,::,1::4])**2,2)
			plt.scatter(k_dist,evals[i], color='k', marker='o', s = np.einsum('ij,j',trial1,decayfunc))
			plt.plot(k_dist,evals[i],linewidth=0.05)
	plt.subplot(1,4,3)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	plt.axhline(y=0,linewidth=0.5, color='k')
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{xz}$ component")
	for i in range(len(evals)):
		if flagarr[i]:
			trial2=np.sum(np.abs(evecs[i,::,2::4])**2,2)
			plt.scatter(k_dist,evals[i], color='k', marker='o', s = np.einsum('ij,j',trial2,decayfunc))
			plt.plot(k_dist,evals[i],linewidth=0.05)
	plt.subplot(1,4,4)
	plt.xlim([k_node[0],k_node[-1]])
	plt.ylim(yrange)
	plt.xticks(k_node,label)
	plt.axhline(y=0,linewidth=0.5, color='k')
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("$d_{xy}$ component")
	for i in range(len(evals)):
		if flagarr[i]:
			trial3=np.sum(np.abs(evecs[i,::,3::4])**2,2)
			plt.scatter(k_dist,evals[i], color='k', marker='o', s = np.einsum('ij,j',trial3,decayfunc))
			plt.plot(k_dist,evals[i],linewidth=0.05)
	plt.tight_layout()
	plt.show(block=False)

def plotsurfacematrixelem(evals,evecs,k_vec,k_dist,k_node,yrange,decaylength=1.0):
	label=(r' ',r'$\Gamma$',r'$\rightarrow M$')
	zlength=len(evecs[1,1])/4
	decayfunc=np.exp(-np.arange(zlength)/decaylength)
	decayfunc=decayfunc/np.sqrt(np.sum(decayfunc**2))*20.
	weight=np.empty(int(zlength*8))
	plt.figure(figsize=(10,5))
	for orb in range(4):
		plt.subplot(1,4,orb+1)
		plt.xlim([k_node[0],k_node[-1]])
		plt.ylim(yrange)
		plt.xticks(k_node, label)
		plt.axhline(y=0,linewidth=0.5, color='k')
		for n in range(len(k_node)):
			plt.axvline(x=k_node[n],linewidth=0.5, color='k')
		plt.title((r"$p_z$ component",r"$d_{yz}$ component",r"$d_{xz}$ component",r"$d_{xy}$ component")[orb])
		for i in range(len(evals)):
			for id, wf in enumerate(evecs[i]):
				weight[id]=np.abs(decayfunc @ (wf.flatten()[2*orb::8]))**2 + np.abs(decayfunc @ (wf.flatten()[2*orb+1::8]))**2
			plt.plot(k_dist,evals[i],linewidth=0.05)
			plt.scatter(k_dist,evals[i], color='k', marker='o', s = weight)
	plt.tight_layout()
	plt.show(block=False)

def getParityXZMirror(eigenfunc):
	Ipauli=	np.array((
						((0., 1.j), (1.j, 0.)),
						((0., 1.), (-1., 0.)),
						((1.j, 0.), (0., -1.j))
					))
	Piy=np.kron(np.diag(np.tile([1.,-1.],int(len(eigenfunc)/4))),Ipauli[1])
	return np.conj(eigenfunc) @ Piy @ eigenfunc

def getFouriercomponent(eigenfunc,kz,decaylength):
	temp=np.exp(1.0j*np.arange(40)*(2*np.pi*kz+1.0j/decaylength))
	planewave=np.empty(8*len(temp))
	for id in range(8):
		planewave[id::8]=temp
	return np.conjugate(planewave) @ eigenfunc

def getParityXZMirrorSpatial(eigenfunc):
	Piy=np.diag(np.tile([1.,1.,-1.,-1.],int(len(eigenfunc)/4)))
	return np.conj(eigenfunc) @ Piy @ eigenfunc

def plotwavefunc(eigenfunc,kparallel,kzflag=False,decaylength=2.0,kzrange=np.linspace(0.,1.0,40)):
	plt.figure(figsize=(10.8,7.2))
	plt.subplot(5,2,1)
	plt.title(r"$p_z$")
	plt.axvline(x=np.argmax(np.abs(eigenfunc[2*0::8]+eigenfunc[2*0+1::8])),linewidth=1., color='r')
	plt.plot(np.real(eigenfunc[2*0::8]),markersize=5.,marker='.',markevery=(0,4),label=r"$\uparrow Re $")
	plt.plot(np.real(eigenfunc[2*0+1::8]),markersize=5.,marker='.',markevery=(1,4),label=r"$\downarrow Re $")
	plt.plot(np.imag(eigenfunc[2*0::8]),markersize=5.,marker='.',markevery=(2,4),label=r"$\uparrow Im $")
	plt.plot(np.imag(eigenfunc[2*0+1::8]),markersize=5.,marker='.',markevery=(3,4),label=r"$\downarrow Im $")
	plt.legend(loc=5)
	plt.subplot(5,2,3)
	plt.title(r"$d_{yz}$")
	plt.axvline(x=np.argmax(np.abs(eigenfunc[2*1::8]+eigenfunc[2*1+1::8])),linewidth=1., color='r')
	plt.plot(np.real(eigenfunc[2*1::8]),markersize=5.,marker='.',markevery=(0,4),label=r"$\uparrow Re $")
	plt.plot(np.real(eigenfunc[2*1+1::8]),markersize=5.,marker='.',markevery=(1,4),label=r"$\downarrow Re $")
	plt.plot(np.imag(eigenfunc[2*1::8]),markersize=5.,marker='.',markevery=(2,4),label=r"$\uparrow Im $")
	plt.plot(np.imag(eigenfunc[2*1+1::8]),markersize=5.,marker='.',markevery=(3,4),label=r"$\downarrow Im $")
	plt.legend(loc=5)
	plt.subplot(5,2,5)
	plt.title(r"$d_{xz}$")
	plt.axvline(x=np.argmax(np.abs(eigenfunc[2*2::8]+eigenfunc[2*2+1::8])),linewidth=1., color='r')
	plt.plot(np.real(eigenfunc[2*2::8]),markersize=5.,marker='.',markevery=(0,4),label=r"$\uparrow Re $")
	plt.plot(np.real(eigenfunc[2*2+1::8]),markersize=5.,marker='.',markevery=(1,4),label=r"$\downarrow Re $")
	plt.plot(np.imag(eigenfunc[2*2::8]),markersize=5.,marker='.',markevery=(2,4),label=r"$\uparrow Im $")
	plt.plot(np.imag(eigenfunc[2*2+1::8]),markersize=5.,marker='.',markevery=(3,4),label=r"$\downarrow Im $")
	plt.legend(loc=5)
	plt.subplot(5,2,7)
	plt.title(r"$d_{xy}$")
	plt.axvline(x=np.argmax(np.abs(eigenfunc[2*3::8]+eigenfunc[2*3+1::8])),linewidth=1., color='r')
	plt.plot(np.real(eigenfunc[2*3::8]),markersize=5.,marker='.',markevery=(0,4),label=r"$\uparrow Re $")
	plt.plot(np.real(eigenfunc[2*3+1::8]),markersize=5.,marker='.',markevery=(1,4),label=r"$\downarrow Re $")
	plt.plot(np.imag(eigenfunc[2*3::8]),markersize=5.,marker='.',markevery=(2,4),label=r"$\uparrow Im $")
	plt.plot(np.imag(eigenfunc[2*3+1::8]),markersize=5.,marker='.',markevery=(3,4),label=r"$\downarrow Im $")
	plt.legend(loc=5)
	plt.subplot(5,2,9)
	plt.title(r"$\langle y \rightarrow -y\rangle $")
	plt.plot(np.abs(eigenfunc[::8])**2+np.abs(eigenfunc[1::8])**2+np.abs(eigenfunc[4::8])**2+np.abs(eigenfunc[5::8])**2 - np.abs(eigenfunc[2::8])**2 - np.abs(eigenfunc[3::8])**2 - np.abs(eigenfunc[6::8])**2 - np.abs(eigenfunc[7::8])**2)
	if kzflag:
		eigenkz=np.empty(len(kzrange)*8)
		zlength=int(round(len(eigenfunc)/8,None))
		decayfunc=np.exp(-1.0*np.arange(zlength)/decaylength)
		decayfunc/=np.sqrt(np.sum(decayfunc))
		for id,kz in enumerate(kzrange):
			planewave=np.exp(-1.0j*np.arange(zlength)*(2*np.pi*kz))*decayfunc
			for n in range(8):
				eigenkz[8*id+n] = planewave @ eigenfunc[n::8]
		plt.subplot(5,2,2)
		plt.title(r"$| \sum_z \psi_{p_z \sigma}(z) e^{(ik_z-1/\lambda) z}|^2$ for $k_\parallel=$"+str(kparallel)+r"$\lambda=$"+str(decaylength))
		plt.xlabel(r"$k_z c/2\pi$")
		plt.plot(kzrange,np.abs(eigenkz[1-1::8])**2,markersize=5.,marker='.',markevery=(0,2),label=r"$\uparrow$")
		plt.plot(kzrange,np.abs(eigenkz[2-1::8])**2,markersize=5.,marker='.',markevery=(1,2),label=r"$\downarrow$")
		plt.legend(loc=5)
		plt.subplot(5,2,4)
		plt.title(r"$| \sum_z \psi_{d_{yz} \sigma}(z) e^{(ik_z-1/\lambda) z}|^2$ for $k_\parallel=$"+str(kparallel)+r"$\lambda=$"+str(decaylength))
		plt.xlabel(r"$k_z c/2\pi$")
		plt.plot(kzrange,np.abs(eigenkz[3-1::8])**2,markersize=5.,marker='.',markevery=(0,2),label=r"$\uparrow$")
		plt.plot(kzrange,np.abs(eigenkz[4-1::8])**2,markersize=5.,marker='.',markevery=(1,2),label=r"$\downarrow$")
		plt.legend(loc=5)
		plt.subplot(5,2,6)
		plt.title(r"$| \sum_z \psi_{d_{xz} \sigma}(z) e^{(ik_z-1/\lambda) z}|^2$ for $k_\parallel=$"+str(kparallel)+r"$\lambda=$"+str(decaylength))
		plt.xlabel(r"$k_z c/2\pi$")
		plt.plot(kzrange,np.abs(eigenkz[5-1::8])**2,markersize=5.,marker='.',markevery=(0,2),label=r"$\uparrow$")
		plt.plot(kzrange,np.abs(eigenkz[6-1::8])**2,markersize=5.,marker='.',markevery=(1,2),label=r"$\downarrow$")
		plt.legend(loc=5)
		plt.subplot(5,2,8)
		plt.title(r"$| \sum_z \psi_{d_{xy} \sigma}(z) e^{(ik_z-1/\lambda) z}|^2$ for $k_\parallel=$"+str(kparallel)+r"$\lambda=$"+str(decaylength))
		plt.xlabel(r"$k_z c/2\pi$")
		plt.plot(kzrange,np.abs(eigenkz[7-1::8])**2,markersize=5.,marker='.',markevery=(0,2),label=r"$\uparrow$")
		plt.plot(kzrange,np.abs(eigenkz[8-1::8])**2,markersize=5.,marker='.',markevery=(1,2),label=r"$\downarrow$")
		plt.legend(loc=5)
	plt.tight_layout()
	plt.show(block=False)

def findpeakZ(eigenfunc):
	return [np.argmax(np.abs(eigenfunc[2*0::8]+eigenfunc[2*0+1::8])),np.argmax(np.abs(eigenfunc[2*1::8]+eigenfunc[2*1+1::8])),np.argmax(np.abs(eigenfunc[2*2::8]+eigenfunc[2*2+1::8])),np.argmax(np.abs(eigenfunc[2*3::8]+eigenfunc[2*3+1::8]))]

def peakvariation(paramrange,xlabel,peakpos):
	# peakpos=np.zeros((4,len(paramrange))
	plt.figure()
	plt.title(r"Layer on which Dirac state peaks")
	plt.xlabel(xlabel)
	for id in range(len(paramrange)):
		FST_bulk = setup_model3d(scalefactor1=paramrange[id],Lsymbr=0.000000,LR=0.000000)
		FST_slab=setup_model2d(FST_bulk,40)
		potential=np.zeros(4*40)
		for z in np.arange(40):
			potential[4*z:4*z+3]=0.000001*z
		FST_slab.set_onsite(potential,mode="add")
		evecs=FST_slab.solve_one([-0.000125, 0.],eig_vectors=True)[1]
		peakpos[:,id]=findpeakZ(evecs[78].flatten())
		# sys.stdout = open(os.devnull, 'w')
		# (k_vec,k_dist,k_node) = symGMcut3d(FST_slab,0.005,81)
		# sys.stdout = sys.__stdout__
		# (evals,evecs)=FST_slab.solve_all(k_vec,eig_vectors=True)
		# peakpos[:,id]=findpeakZ(evecs[78,39].flatten())
	plt.plot(paramrange,peakpos[0],marker="o",label=r"$p_z$")
	plt.plot(paramrange,peakpos[1],marker="*",label=r"$d_{yz}$")
	plt.plot(paramrange,peakpos[2],marker="x",label=r"$d_{xz}$")
	plt.plot(paramrange,peakpos[3],marker="+",label=r"$d_{xy}$")
	plt.legend()
	plt.tight_layout()
	plt.show(block=False)

def plotXZparitybands(yrange,emin,emax,kindex=18):
	pauli=	np.array((
						((0., 1.), (1., 0.)),
						((0., -1.j), (1.j, 0.)),
						((1., 0.), (0., -1.))
					))
	Piy=np.kron(np.diag(np.tile([1.,-1.],int(len(evecs[1,1])/2))),pauli[1])
	colors=np.tile(['r','r','b','b'],int(len(evals)/4))
	label=(r' ',r'$\bar{\Gamma}$',r'$\rightarrow \bar{M}$')
	plt.figure(figsize=(10,7))
	plt.subplot(1,4,1)
	plt.xlim([k_node[0],k_node[-1]])
	plt.xticks(k_node, label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n],linewidth=0.5, color='k')
	plt.title("Parity")
	for i in np.arange(emin,emax):
		Parity=np.zeros(len(evecs[1]))
		for k in np.arange(len(evecs[i])):
			eigvectemp=np.reshape(evecs[i,k],len(evecs[1,1])*2)
			Parity[k]=np.conj(eigvectemp) @ Piy @ eigvectemp
		# Parity=np.sum(np.conjugate(evecs[i])*np.tile([1.,-1.],int(len(evecs[1,1])/2))[None,:,None]*np.einsum('ni,kli',pauli[2],evecs[i]),(1,2))
		plt.scatter(k_dist, (np.zeros(len(Parity))+Parity), marker='o', color=colors[i])
	plt.subplot(1,4,2)
	plt.xlim([k_node[0],k_node[-1]])
	plt.xticks(k_node,label)
	for n in range(len(k_node)):
		plt.axvline(x=k_node[n], color='k')
	plt.title("Bands")
	for i in np.arange(emin,emax):
		plt.scatter(k_dist,evals[i], marker='o',color=colors[i])
		plt.plot(k_dist,evals[i],linewidth=0.1)
	plt.subplot(1,4,3)
	plt.title("Wavefunction for k="+str(k_vec[18]))
	plt.plot(np.abs(evecs[i,18].flatten()))
	plt.subplot(1,4,4)
	plt.title("Wavefunction for k="+str(k_vec[22]))
	plt.plot(np.abs(evecs[i,22].flatten()))
	plt.tight_layout()
	plt.show(block=False)

def plotXZparitywithzDependence(evals,evecs,k_vec,k_dist,k_node,eigindexrange,koffset=2):
	eigcount=len(eigindexrange)
	count=0
	# colors=(['r','b'])
	energymin=min(evals[eigindexrange[1]:eigindexrange[-1]].flatten())
	energymax=max(evals[eigindexrange[1]:eigindexrange[-1]].flatten())
	kmidindex=int((len(k_vec)-1)/2)
	label=(r' ',r'$\bar{\Gamma}$',r'$\rightarrow \bar{M}$')
	plt.figure(figsize=(10,7))
	for en in eigindexrange:
		plt.subplot(eigcount,3,3*count+1)
		plt.title(r"Wavefunction for k=$\bar{\Gamma}-$"+str(koffset))
		plt.plot(np.abs(evecs[en,kmidindex-koffset,0::4].flatten()),label=r"p_z")
		plt.plot(np.abs(evecs[en,kmidindex-koffset,1::4].flatten()),label=r"d_{yz}")
		plt.plot(np.abs(evecs[en,kmidindex-koffset,2::4].flatten()),label=r"d_{xz}")
		plt.plot(np.abs(evecs[en,kmidindex-koffset,3::4].flatten()),label=r"d_{xy}")
		plt.legend()
		plt.subplot(eigcount,3,3*count+2)
		plt.xlim([(k_node[0]+k_node[1])/2,(k_node[1]+k_node[-1])/2])
		plt.xticks(k_node, label)
		plt.ylim(energymin-0.01,energymax+0.01)
		for n in range(len(k_node)):
			plt.axvline(x=k_node[n],linewidth=0.5, color='k')
		plt.title("Band #"+str(en)+" weighted by space-only XZ-Mirror eigenvalue")
		# for i in range(len(evals)):
			# plt.plot(k_dist,evals[i],linewidth=0.5)
		for k in np.arange(len(evecs[1])):
			# plt.scatter(k_dist[k],evals[en,k], marker='o', c=colors[int(round(np.real(getParityXZMirror(evecs[en,k].flatten())/2+0.5),None))])
			plt.scatter(k_dist[k],evals[en,k], marker='o', s=20.*np.real(getParityXZMirrorSpatial(evecs[en,k].flatten())/2+0.5))
		# plt.scatter(k_dist[kmidindex-koffset],evals[en,kmidindex-koffset],c='k',marker='o')
		# plt.scatter(k_dist[kmidindex+koffset],evals[en,kmidindex+koffset],c='k',marker='o')
		plt.subplot(eigcount,3,3*count+3)
		plt.title(r"Wavefunction for k=$\bar{\Gamma}+$"+str(koffset))
		plt.plot(np.abs(evecs[en,kmidindex+koffset,0::4].flatten()),label=r"p_z")
		plt.plot(np.abs(evecs[en,kmidindex+koffset,1::4].flatten()),label=r"d_{yz}")
		plt.plot(np.abs(evecs[en,kmidindex+koffset,2::4].flatten()),label=r"d_{xz}")
		plt.plot(np.abs(evecs[en,kmidindex+koffset,3::4].flatten()),label=r"d_{xy}")
		plt.legend()
		count+=1
	plt.tight_layout()
	plt.show(block=False)		

def plotXZparitywithzDecayFunc(evals,evecs,k_vec,k_dist,k_node,eigindexrange,koffset=2):
	zlength=len(evecs[1,1])/4
	decaylength=1.0
	decayfunc=np.exp(-np.arange(zlength)/decaylength)
	eigcount=len(eigindexrange)
	count=0
	energymin=min(evals[eigindexrange[1]:eigindexrange[-1]].flatten())
	energymax=max(evals[eigindexrange[1]:eigindexrange[-1]].flatten())
	kmidindex=int((len(k_vec)-1)/2)
	label=(r' ',r'$\bar{\Gamma}$',r'$\rightarrow \bar{M}$')
	plt.figure(figsize=(10,7))
	for en in eigindexrange:
		plt.subplot(eigcount,3,3*count+1)
		plt.title(r"Wavefunction for k=$\bar{\Gamma}-$"+str(koffset))
		plt.plot(np.abs(evecs[en,kmidindex-koffset].flatten()))
		plt.subplot(eigcount,3,3*count+2)
		plt.xlim([(k_node[0]+k_node[1])/2,(k_node[1]+k_node[-1])/2])
		plt.xticks(k_node, label)
		plt.ylim(energymin-0.001,energymax+0.001)
		for n in range(len(k_node)):
    			plt.axvline(x=k_node[n],linewidth=0.5, color='k')
		plt.title("Band #"+str(en))
		# +" colored by XZ-Mirror eigenvalue")
		for k in np.arange(len(evecs[1])):
			plt.scatter(k_dist[k],evals[en,k], marker='o', c='g', s=10.*np.real(getParityXZMirror(np.kron(decayfunc,np.ones(8))*evecs[en,k].flatten())/2+0.5))
		# plt.scatter(k_dist[kmidindex-koffset],evals[en,kmidindex-koffset],c='k',marker='o')
		# plt.scatter(k_dist[kmidindex+koffset],evals[en,kmidindex+koffset],c='k',marker='o')
		plt.subplot(eigcount,3,3*count+3)
		plt.title(r"Wavefunction for k=$\bar{\Gamma}+$"+str(koffset))
		plt.plot(np.abs(evecs[en,kmidindex+koffset].flatten()))
		count+=1
	plt.tight_layout()
	plt.show(block=False)		

def checkoutall(kxmax=0.02,kxlength=41,kzmax=0.1,kzlength=50,kxforkzplot=0.0,enviewrange=[-0.06,0.05],zlength=40,scalefactorC=1.,scalefactorX=1.,scalefactorhopp=1.,scalefactorhopd=1.,scalefactorgap=1.,scalefactordxytop=1.,M02=-0.012,M12=-3.8,M22=0.00425,M01=0.002,M11=2.4,M21=-0.006,M03=-0.002,M13=-0.2,M23=0.0001,	A = -4.4*1., B = 2.2*1., C = 0.004*1., X = 0.412*1., L1=0.01,L2=0.001,L3=0.004,decaylength=1.,kxplotflag=True,kzplotflag=True,wfplotflag=True,note="",markerareascale=5.):
	# FST_bulk=setup_model3dasbefore(scalefactorC,scalefactorX,scalefactorhopp,scalefactorhopd,scalefactorgap,scalefactordxytop)
	FST_bulk=setup_model3d(M02,M12,M22,M01,M11,M21,M03,M13,M23,A,B,C,X,L1,L2,L3)
	FST_slab=setup_model2d(FST_bulk,zlength)
	FST_slab.set_onsite(np.kron(np.arange(zlength)*0.000001,np.array([1.0,1.0,1.0,1.0])),mode="add")
	(k_vec,k_dist,k_node)=symGMcut3d(FST_slab,kmax=kxmax,klength=kxlength)
	(evals,evecs)=FST_slab.solve_all(k_vec,eig_vectors=True)
	if kxplotflag:
		plotsurfacebands(evals,markerareascale*evecs,k_vec,k_dist,k_node,enviewrange,decaylength)
		_=plt.title(note)
		show()
	if wfplotflag:
		plotwavefunc(evecs[2*zlength-2,int((kxlength-1)/2-1)].flatten(),k_vec[int((kxlength-1)/2-1)],kzflag=True)
		_=plt.title(note)
		show()
	if kzplotflag:
		kzarr=np.linspace(0.,kzmax,kzlength)
		plt.figure()
		colors=np.tile(['r','r','b','b','g','g','k','k'],50)
		dataeigenvals=np.empty((8,len(kzarr)))
		FST_bulk=setup_model3d()
		for id,kz in enumerate(kzarr):
			dataeigenvals[:,id]=FST_bulk.solve_one([kxforkzplot,0.,kz])
		for id,band in enumerate(dataeigenvals):
			plt.plot(kzarr,band,'.',c=colors[id])
		FST_bulk=setup_model3d(M02,M12,M22,M01,M11,M21,M03,M13,M23,A,B,C,X,L1,L2,L3)
		for id,kz in enumerate(kzarr):
			dataeigenvals[:,id]=FST_bulk.solve_one([kxforkzplot,0.,kz])
		for id,band in enumerate(dataeigenvals):
			plt.plot(kzarr,band,'--',c=colors[id])
		_=plt.title(note)
		show()