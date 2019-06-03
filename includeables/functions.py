from __future__ import print_function
from pythtb import tb_model # import TB model class
import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit import minimize, Parameters, report_fit

def show():
	plt.show(block=False)

def model1d(ep,ed,tzp,tzd,l1,l1z,l3,mp=1000.,md1=1000.,md2=1000.,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txy=0.,txydiag=0.,X=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.,kx=0.,ky=0.):
	my_model=tb_model(dim_r=1,dim_k=1,lat=[[1.0]],orb=[[0.],[0.],[0.]],nspin=2)
	
	### diagonal terms
	#(1+kx^2+ky^2)
	my_model.set_onsite([ep,ed,ed])
	my_model.set_onsite([(kx**2+ky**2)/(2*mp),kx**2/md1+ky**2/md2,kx**2/md2+ky**2/md1],mode="add")
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
	my_model.set_hop([0., lWeyl*kx*1.0j, lWeyl*ky*1.0j,0.],1,2,[1.],mode="add")
	my_model.set_hop([0.,-lWeyl*kx*1.0j,-lWeyl*ky*1.0j,0.],1,2,[-1.],mode="add")
	
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

def model1dcut(nsites=40,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,mp=0.,md1=0.,md2=0.,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txy=0.,txydiag=0.,X=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.,kx=0.,ky=0.):
	return model1d(ep,ed,tzp,tzd,l1,l1z,l3,mp,md1,md2,tdiagp,tdiagd1,tdiagd2,txy,txydiag,X,Xz,l1plane,l1diag,lWeyl,l3plane1,l3plane2,l3xy,l3x2y2,kx,ky).cut_piece(num=nsites,fin_dir=0,glue_edgs=False)

def calcEnergy1(ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,kz=0.):
	return model1d(ep,ed,tzp,tzd,l1,l1z,l3).solve_one([kz])[0]

def calcEnergy2(ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,kz=0.):
	return model1d(ep,ed,tzp,tzd,l1,l1z,l3).solve_one([kz])[2]

# allows opt='none','dxzweightsAsLineplot','dxzweightsAsMarkersize'
def plotGZ(opt='none',sizescale=100.,bandindices=[0,1,2,3,4,5],kpoints=100,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004):
	plt.rc('text', usetex=True)
	plt.rcParams['axes.linewidth'] = 2.
	plt.rc('font', family='serif', serif='Times New Roman',weight='bold')
	plt.figure(figsize=(7,7.5))
	plt.axhline(linestyle='--',color='k')
	plt.ylabel(r"E (meV)", fontsize=25, color='k', x=2.0)
	plt.xlabel(r'$k_\parallel ($\AA$^{-1}$)', fontsize=25, color='k', x=2.0)
	FST_bulk=model1d(ep,ed,tzp,tzd,l1,l1z,l3)
	mask=np.diag(np.array([0.,0.,0.,0.,1.,1.]))
	# mask=np.diag(np.array([1.,1.,0.,0.,0.,0.]))
	kzarr=np.linspace(0.,1.,num=kpoints)
	energies=np.empty((6,kzarr.shape[0]))
	dxzweights=np.empty((6,kzarr.shape[0]))
	for id,kz in enumerate(kzarr):
		energies[:,id]=FST_bulk.solve_one([kz])
		eigvec=np.reshape(FST_bulk.solve_one([kz],eig_vectors=True)[1],(6,6)).T
		dxzweights[:,id]=np.diag(np.conjugate(eigvec.T) @ mask @ eigvec)
	if opt=='dxzweightsAsLineplot':
		for band in energies:
			plt.plot(kzarr,band*1000.)
		plt.figure()
		for id,wt in enumerate(dxzweights):
			if id in bandindices:
				plt.plot(kzarr,wt)
	elif opt=='dxzweightsAsMarkersize':
		for id,band in enumerate(energies):
			if id in bandindices:
				# plt.plot(kzarr,band*1000.,'k-',lw=2.)
				plt.scatter(kzarr,band*1000.,s=dxzweights[id]*sizescale, marker='o', facecolors='none', edgecolors='r',label=r'$d_{xz}$ weight')
		# plt.legend(loc='best',ncol= 2,fontsize=25,frameon=False)
		plt.title(r'$E(0,0,k_z)$', fontsize=25)
	else:
		for band in energies:
			plt.plot(kzarr,band*1000.)
	plt.tick_params(axis='y', labelcolor='k', labelsize=20, size=4, width=2 )
	plt.tick_params(axis='x', labelsize=20,size=4,width=2)		
	plt.tight_layout()
	show()

#takes dir='GM','GX'
def plotInplane(kz=0.,klim=0.25,yrange=[-80,10],dir='GM',kpoints=100,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,mp=0.1316,md1=-0.109,md2=-2.5,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txy=-4.4,txydiag=0.,X=0.412,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.):
	plt.rc('text', usetex=True)
	plt.rcParams['axes.linewidth'] = 2.
	plt.rc('font', family='serif', serif='Times New Roman',weight='bold')
	plt.figure(figsize=(7,7.5))
	# plt.ylim(yrange)
	plt.axhline(linestyle='--',color='k')
	plt.ylabel(r"E (meV)", fontsize=25, color='k', x=2.0)
	plt.xlabel(r'$k_\parallel ($\AA$^{-1}$)', fontsize=25, color='k', x=2.0)
	kmarr=np.linspace(-klim/(2*0.3/1.65),klim/(2*0.3/1.65),num=kpoints)
	#kmarr=np.linspace(0.,0.363,num=kpoints) maps to 0.3 AngsInv
	energies=np.empty((6,kmarr.shape[0]))
	for id,km in enumerate(kmarr):
		(kx0,ky0)=(0.707*km,0.707*km) if dir=='GM' else (km,0.)
		FST_bulk=model1d(ep,ed,tzp,tzd,l1,l1z,l3,mp,md1,md2,tdiagp,tdiagd1,tdiagd2,txy,txydiag,X,Xz,l1plane,l1diag,lWeyl,l3plane1,l3plane2,l3xy,l3x2y2,kx=kx0,ky=ky0)
		energies[:,id]=FST_bulk.solve_one([kz])
	for band in energies:
		plt.plot(kmarr*2*0.3/1.65,band*1000.,'k-', lw=2.0)
	plt.tick_params(axis='y', labelcolor='k', labelsize=20, size=4, width=2 )
	plt.tick_params(axis='x', labelsize=20,size=4,width=2)
	plt.legend(loc='best',ncol= 2,fontsize=25,frameon=False)
	plt.title(r'E($k_\parallel \in \Gamma M,k_z=$'+str(2*kz)+r'$\pi/c)$', fontsize=25)
	plt.tight_layout()
	show()

#takes dir='GM','GX'
def plotInplaneSurface(finSize=40,kz=0.,ikz=1.,sizescale=100.,klim=0.25,yrange=[-80,10],dir='GM',kpoints=100,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,mp=0.1316,md1=-0.109,md2=-2.5,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txy=-4.4,txydiag=0.,X=0.412,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.):
	plt.rc('text', usetex=True)
	plt.rcParams['axes.linewidth'] = 2.
	plt.rc('font', family='serif', serif='Times New Roman',weight='bold')
	plt.figure(figsize=(7,7.5))
	# plt.ylim(yrange)
	plt.axhline(linestyle='--',color='k')
	plt.ylabel(r"E (meV)", fontsize=25, color='k', x=2.0)
	plt.xlabel(r'$k_\parallel ($\AA$^{-1}$)', fontsize=25, color='k', x=2.0)
	kmarr=np.linspace(-klim/(2*0.3/1.65),klim/(2*0.3/1.65),num=kpoints)
	#kmarr=np.linspace(0.,0.363,num=kpoints) maps to 0.3 AngsInv
	mask=np.empty(finSize*6)
	for z in np.arange(finSize):
		for orb in range(6):
			mask[6*z+orb]=np.exp(1.0j*(z+0.0j)*(kz+ikz*1.0j))*[0.,0.,-1.,-1.,1.,1.][orb]
	energies=np.empty((6*finSize,kmarr.shape[0]))
	dxzweights=np.empty((6*finSize,kmarr.shape[0]))
	for id,km in enumerate(kmarr):
		(kx0,ky0)=(0.707*km,0.707*km) if dir=='GM' else (km,0.)
		FST_surf=model1dcut(finSize,ep,ed,tzp,tzd,l1,l1z,l3,mp,md1,md2,tdiagp,tdiagd1,tdiagd2,txy,txydiag,X,Xz,l1plane,l1diag,lWeyl,l3plane1,l3plane2,l3xy,l3x2y2,kx=kx0,ky=ky0)
		eigensol=FST_surf.solve_one(eig_vectors=True)
		energies[:,id]=eigensol[0]
		eigvec=np.reshape(eigensol[1],(120,120)).T
		dxzweights[:,id]=np.diag(np.conjugate(eigvec.T) @ np.diag(mask) @ eigvec)
	for bid,band in enumerate(energies):
		#plt.plot(kmarr*2*0.3/1.65,band*1000.,'k-', lw=0.2)
		plt.scatter(kmarr*2*0.3/1.65,band*1000.,s=abs(dxzweights[bid]*sizescale), marker='o', facecolors='none', edgecolors='r')
	plt.tick_params(axis='y', labelcolor='k', labelsize=20, size=4, width=2 )
	plt.tick_params(axis='x', labelsize=20,size=4,width=2)
	plt.title(r'E($k_\parallel \in \Gamma M,k_z=$'+str(2*kz)+r'$\pi/c)$', fontsize=25)
	plt.tight_layout()
	show()

plotGZ(kpoints=500,l1z=-0.0005)
plotGZ(opt='dxzweightsAsMarkersize',sizescale=10,kpoints=500,ep=0.04,ed=-0.00,tzp=0.16,tzd=-0.28,l1=0.0105,l1z=0.,l3=0.019)
plotInplane(kz=0.5,kpoints=500,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,mp=1.8,md1=-1.8,md2=-1.8,txy=-0.75,X=0*0.412,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txydiag=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.)
plotInplane(kz=0.5,kpoints=100,ep=0.04,ed=-0.00,tzp=0.16,tzd=-0.28,l1=0.0105,l1z=0.,l3=0.019,mp=0.20833,md1=-0.3125,md2=-0.0833,txy=-4.4,X=0.412,tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txydiag=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.)
plotGZ(opt='dxzweightsAsMarkersize',sizescale=100.)
plotGZ(opt='dxzweightsAsLineplot',bandindices=[0,1])

plotInplaneSurface(finSize=20,dir='GX',kpoints=500,sizescale=1000.,
ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,mp=1.8,md1=-1.8,md2=-1.8,txy=-0.75,X=0*0.412,
tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txydiag=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.)

plotInplaneSurface(finSize=20,dir='GX',klim=0.05,ikz=100.,kpoints=300,sizescale=1000.,
ep=0.04,ed=-0.00,tzp=0.16,tzd=-0.28,l1=0.0105,l1z=0.,l3=0.019*2,mp=0.20833,md1=-0.3125,md2=-0.0833,txy=-4.4,X=0.412,
tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txydiag=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.)

plotGZ(opt='dxzweightsAsMarkersize',sizescale=10,kpoints=500,
ep=0.04/20-.003,ed=-0.00-.003,tzp=0.16/40*2.5,tzd=-0.28/40,l1=0.0105/20*20,l1z=0.,l3=0.019/20*5)

plotGZ(sizescale=10,kpoints=500,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004)

plotInplane(kz=0.5,kpoints=100,klim=0.06,dir='GX',
ep=0.04/20-.003,ed=-0.00-.003,tzp=0.16/40*2.5,tzd=-0.28/40,l1=0.0105/20*20,l1z=0.,l3=0.019/20*5,mp=0.20833*13,md1=-0.3125*13,md2=-0.0833*13,txy=-4.4/20,X=0.412/20,
tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txydiag=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.)

plotInplaneSurface(finSize=20,dir='GX',klim=0.06,kz=0.,ikz=0.3,kpoints=300,sizescale=1000.,
ep=0.04/20-.003,ed=-0.00-.003,tzp=0.16/40*2.5,tzd=-0.28/40,l1=0.0105/20*20,l1z=0.,l3=0.019/20*5,mp=0.20833*13,md1=-0.3125*13,md2=-0.0833*13,txy=-4.4/20,X=0.412/20,
tdiagp=0.,tdiagd1=0.,tdiagd2=0.,txydiag=0.,Xz=0.,l1plane=0.,l1diag=0.,lWeyl=0.,l3plane1=0.,l3plane2=0.,l3xy=0.,l3x2y2=0.)

FST_surf=model1dcut(nsites=40,ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,kx=0.,ky=0.,mp=1.8,md1=-1.8,md2=-1.8)
FST_bulk=model1d(ep=0.,ed=-0.002,tzp=0.01,tzd=-0.00425,l1=0.01,l1z=-0.0005,l3=0.004,kx=0.,ky=0.,mp=1.8,md1=-1.8,md2=-1.8)
np.reshape(FST_surf.solve_one(eig_vectors=True)[1],(240,240)).shape

#fitting ARPES peaks to the kz dispersion of this model... bad idea because of overfitting
datasin=np.sin(np.linspace(0,2*np.pi,20))+np.random.rand(20)/100.
def resid(params,x,dataarr):
	a=params['a'].value
	return np.sin(a*x)-dataarr

params=Parameters()
params.add('a',value=0.4)
result=minimize(resid,params,args=(np.linspace(0,2*np.pi,20),datasin))
report_fit(result.params)
result.params['a'].value

resid(params,np.linspace(0,2*np.pi,20),datasin)
plt.figure()
plt.scatter(np.linspace(0,2*np.pi,20),datasin)
plt.plot(np.linspace(0,np.pi*2,20),np.sin(result.params['a'].value*np.linspace(0,np.pi*2,20)))
show()

os.chdir('D://Dropbox/Iron Based SC/ARPES Peaks Data')
data1=np.loadtxt('alpha1diam.dat',usecols=(0,1,2))
data2=np.loadtxt('alpha2diam.dat',usecols=(0,1,2))

plt.figure()
plt.scatter(data1[:,0],data1[:,1])
show()

plt.figure()
plt.scatter(data2[:,0],data2[:,1])
show()

params=Parameters()
params.add('ep',value=0.04/20-.003)
params.add('ed',value=-0.00-.003)
params.add('tzp',value=0.16/40*2.5)
params.add('tzd',value=-0.28/40)
params.add('l1',value=0.0105/20*20)
params.add('l1z',value=0.)
params.add('l3',value=0.019/20*5)

def calcdataset1(params,x):
	ep=params['ep'].value
	ed=params['ed'].value
	tzp=params['tzp'].value
	tzd=params['tzd'].value
	l1=params['l1'].value
	l1z=params['l1z'].value
	l3=params['l3'].value
	data=np.copy(x)
	for id,k in enumerate(x):
		data[id]=calcEnergy1(ep,ed,tzp,tzd,l1,l1z,l3,k)
	return data

def calcdataset2(params,x):
	ep=params['ep'].value
	ed=params['ed'].value
	tzp=params['tzp'].value
	tzd=params['tzd'].value
	l1=params['l1'].value
	l1z=params['l1z'].value
	l3=params['l3'].value
	data=np.copy(x)
	for id,k in enumerate(x):
		data[id]=calcEnergy2(ep,ed,tzp,tzd,l1,l1z,l3,k)
	return data

def residual(params, x1, x2, data1, data2):
	return np.concatenate((calcdataset1(params,x1/2)-data1,calcdataset2(params,x2/2)-data2),axis=0)

result=minimize(residual,params,args=(data1[:,0],data2[:,0],data1[:,1],data2[:,1]))
report_fit(result.params)

plt.figure()
plt.scatter(data1[:,0],data1[:,1])
plt.plot(np.linspace(data1[1,0],data1[-1,0],100),calcdataset1(result.params,1/2*np.linspace(data1[1,0],data1[-1,0],100)))
show()

plt.figure()
plt.scatter(data2[:,0],data2[:,1])
plt.plot(np.linspace(data2[1,0],data2[-1,0],100),calcdataset2(result.params,1/2*np.linspace(data2[1,0],data2[-1,0],100)))
show()