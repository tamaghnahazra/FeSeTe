from __future__ import print_function
from pythtb import tb_model # import TB model class
import numpy as np
import matplotlib.pyplot as plt

def show():
	plt.show(block=False)

def model1d(ep,ed,tzp,tzd,l1,l1z,l3,mp=1000.,md1=1000.,md2=1000.,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txy=0.,txydiag=0.,X=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.,kx=0.,ky=0.):
	my_model=tb_model(dim_r=1,dim_k=1,lat=[[1.0]],orb=3,nspin=2)
	
	### diagonal terms
	#(1+kx^2+ky^2)
	my_model.set_onsite([ep,ed,ed])
	my_model.set_onsite([(kx**2+ky**2)/2*mp,kx**2/md1+ky**2/md2,kx**2/md2+ky**2/md1],mode="add")
	#(1+kx^2+ky^2)\cos kz 
	my_model.set_hop(tzp,0,0,[1.])
	my_model.set_hop(tdiagp*(kx**2+ky**2),0,0,[1.],mode="add")
	my_model.set_hop(tzd,1,1,[1.])
	my_model.set_hop(tdiagd1*(kx**2)+tdiagd2*(ky**2),1,1,[1.],mode="add")
	my_model.set_hop(tzd,2,2,[1.])
	my_model.set_hop(tdiagd2*(kx**2)+tdiagd1*(ky**2),2,2,[1.],mode="add")
	
	### dxz dyz terms
	#kx ky
	my_model.set_hop(txy*kx*ky,1,2,[0.])
	my_model.set_hop(txydiag*kx*ky,1,2,[1.],mode="add")
	my_model.set_hop(txydiag*kx*ky,1,2,[-1.],mode="add")
	#\sigma_z (1+kx^2+ky^2)(1+\cos kz)
	my_model.set_hop([0.,0.,0., (l1+l1plane*(kx**2+ky**2))*1.0j], 1, 2, [0.],mode="add")
	my_model.set_hop([0.,0.,0., (l1z+l1diag*(kx**2+ky**2))*1.0j], 1, 2, [1.],mode="add")
	my_model.set_hop([0.,0.,0., (l1z+l1diag*(kx**2+ky**2))*1.0j], 1, 2, [-1.],mode="add")
	#(\sigma_x kx +\sigma_y ky)\sin kz
	my_model.set_hop([0., ldirac*kx*1.0j, lWeyl*ky*1.0j,0.],1,2,[1.],mode="add")
	my_model.set_hop([0.,-ldirac*kx*1.0j,-lWeyl*ky*1.0j,0.],1,2,[-1.],mode="add")
	
	### pz (dyz,dxz) terms
	#(kx,ky)(1+\cos kz)
	my_model.set_hop(X*ky*1.0j,0,1,[0.])
	my_model.set_hop(Xz*ky*1.0j,0,1,[1.],mode="add")
	my_model.set_hop(Xz*ky*1.0j,0,1,[-1.],mode="add")
	my_model.set_hop(X *kx*1.0j,0,2,[0.])
	my_model.set_hop(Xz*kx*1.0j,0,2,[1.],mode="add")
	my_model.set_hop(Xz*kx*1.0j,0,2,[-1.],mode="add")
	#\sigma_z(kx,ky)(1+\cos kz)
	my_model.set_hop([0.,0.,0.,  X*kx*1.0j],0,1,[0.],mode="add")
	my_model.set_hop([0.,0.,0., Xz*kx*1.0j],0,1,[1.],mode="add")
	my_model.set_hop([0.,0.,0., Xz*kx*1.0j],0,1,[-1.],mode="add")
	my_model.set_hop([0.,0.,0.,- X*ky*1.0j],0,2,[0.],mode="add")
	my_model.set_hop([0.,0.,0.,-Xz*ky*1.0j],0,2,[1.],mode="add")
	my_model.set_hop([0.,0.,0.,-Xz*ky*1.0j],0,2,[-1.],mode="add")
	#(\sigma_x,y)\sin kz
	my_model.set_hop([0., l3*1.0j,0.,0.],0,1,[1.],mode="add")
	my_model.set_hop([0.,-l3*1.0j,0.,0.],0,1,[-1.],mode="add")
	my_model.set_hop([0.,0.,-l3*1.0j,0.],0,2,[1.],mode="add")
	my_model.set_hop([0.,0., l3*1.0j,0.],0,2,[-1.],mode="add")
	my_model.set_hop([0., (l3plane1*kx**2+l3plane2*ky**2)*1.0j,0.,0.],0,1,[1.],mode="add")
	my_model.set_hop([0.,-(l3plane1*kx**2+l3plane2*ky**2)*1.0j,0.,0.],0,1,[-1.],mode="add")
	my_model.set_hop([0.,0.,-(l3plane2*kx**2+l3plane1*ky**2)*1.0j,0.],0,2,[1.],mode="add")
	my_model.set_hop([0.,0., (l3plane2*kx**2+l3plane1*ky**2)*1.0j,0.],0,2,[-1.],mode="add")
	#kx ky and kx2-ky2
	my_model.set_hop([0., l3x2y2*(kx**2-ky**2)*1.0j,-l3xy*kx*ky*1.0j,0.],0,1,[1.],mode="add")
	my_model.set_hop([0.,-l3x2y2*(kx**2-ky**2)*1.0j, l3xy*kx*ky*1.0j,0.],0,1,[1.],mode="add")
	my_model.set_hop([0., l3xy*kx*ky*1.0j, l3x2y2*(kx**2-ky**2)*1.0j,0.],0,2,[1.],mode="add")
	my_model.set_hop([0.,-l3xy*kx*ky*1.0j,-l3x2y2*(kx**2-ky**2)*1.0j,0.],0,2,[1.],mode="add")
	return my_model

def model1dcut(ep,ed,tzp,tzd,l1,l1z,l3,nsites,mp=0.,md1=0.,md2=0.,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txy=0.,txydiag=0.,X=0.,Xz=0.,l1plane=0.,l1diag=0.,ldirac=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.,kx=0.,ky=0.):
	return model1d(ep,ed,tzp,tzd,l1,l1z,l3,mp,md1,md2,tdiagp,tdiagd1,tdiagd2,txy,txydiag,X,Xz,l1plane,l1diag,ldirac,l3plane1,l3plane2,l3xy,l3x2y2,kx,ky).cut_piece(num=nsites,fin_dir=[0],glue_edgs=False)

def plotGZ(kpoints=100,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=0.,l3=0.004):
	FST_bulk=model1d(ep,ed,tzp,tzd,l1,l1z,l3)
	plt.figure()
	kzarr=np.linspace(0.,1.,num=kpoints)
	energies=np.empty((6,kzarr.shape[0]))
	for id,kz in enumerate(kzarr):
		energies[:,id]=FST_bulk.solve_one([kz])
	for band in energies:
		plt.plot(kzarr,band)
	show()

plotGZ(kpoints=500)