import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

def cheb(N):
    if N == 0:
        D = 0; x = 1;
    else:
        x = np.cos(np.pi*np.array(range(0,N+1))/N).reshape([N+1,1])
        c = np.ravel(np.vstack([2, np.ones([N-1,1]), 2])) \
            *(-1)**np.ravel(np.array(range(0,N+1)))
        c = c.reshape(c.shape[0],1)
        X = np.tile(x,(1,N+1))
        dX = X-(X.conj().transpose())
        D  = (c*(1/c).conj().transpose())/(dX+(np.eye(N+1)))   # off-diagonal entries
        D  = D - np.diag(np.sum(D,1))   # diagonal entries
    return D,x

def build_A_start(Ny,U,g0,dU,f0,H,HDy,dH):

    Ap = PETSc.Mat().create()
    Ap.setSizes([3*Ny+1, 3*Ny+1]); Ap.setFromOptions(); Ap.setUp()
    start,end = Ap.getOwnershipRange()

    # Set up all parts of A except Blocks A10 and A12 (set up the ones that don't use kx)
    for i in range(start,end):
        if (0 <= i <= Ny): #first row; Ny+1
            Ap[i,i] = U[i]
            Ap[i, 2*Ny+i] = g0 #Block A02
            if (1 <= i <= Ny-1): # top row and bottom of block is all 0 for A01
                Ap[i, Ny+i] = dU[i] - f0 #Block A01

        if Ny <= i <= 2*Ny: #2nd row; Ny-1
            Ap[i,i] = U[i-Ny]

        if 2*Ny <= i <= 3*Ny: #3rd row
            cols1 = i-2*Ny  #1 - Ny
            cols2 = i-Ny+1  #Ny+1 - 2*Ny
            Ap[i,i] = U[cols1]
            Ap[i,cols1] = H[cols1] #Block A20
            #Block A21
            if (cols1 == 0): #assign only 1 value (top corner of block)
                Ap[i,cols2] = HDy[0,1]
            elif (cols1 == Ny): #bottom corner
                Ap[i,cols2-2] = HDy[Ny, Ny-1]
            elif (cols1 == 1): #2 elements top
                Ap[i,[cols2-1,cols2]] = [dH[1], HDy[1,2]]
            elif (cols1 == Ny-1): #2 elements bottom
                if sp.issparse(HDy):
                    Ap[i,[cols2-2,cols2-1]] = [0,dH[Ny-1]] + HDy[Ny-1,Ny-2:Ny].todense()
                else:
                    Ap[i,[cols2-2,cols2-1]] = [0,dH[Ny-1]] + HDy[Ny-1,Ny-2:Ny]
            elif (2 <= cols1 <= Ny-2): #3 elements
                Ap[i,cols2-2:cols2+1] = [HDy[cols1, cols1-1], dH[cols1], HDy[cols1, cols1+1]]
    Ap.assemble()
    return Ap

def build_A(Ap,Ny,g0,f0,k2,Dy):
    A = PETSc.Mat().createAIJ([3*Ny+1,3*Ny+1])
    A.setFromOptions(); A.setUp()
    A = Ap.copy()
    start,end = A.getOwnershipRange()

    for i in range(start,end):
        if Ny+1 <= i < 2*Ny:
            cols1 = i-Ny # goes from 1-Ny
            cols3 = i+Ny-1 #goes from Ny+1 :
            A[i,cols1] = -f0/k2 # Block A10
            if sp.issparse(Dy):
                A[i,cols3:cols3+3] = (-g0/k2)*Dy[cols1,cols1-1:cols1+2].todense() # Block A12
            else:
                A[i,cols3:cols3+3] = (-g0/k2)*Dy[cols1,cols1-1:cols1+2]
    A.assemble()
    return A

def solve_eigensystem(A,kx,Ny,nEV,grow,freq,mode,guess):
    t0 = time.time()
    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
    sinv = E.getST()
    E.setOperators(A); E.setDimensions(nEV, SLEPc.DECIDE)
    # E.setTarget(guess)
    # E.setType(SLEPc.EPS.Type.LAPACK)
    # E.setBalance(2) #SLEPc.EPS.Balance.ONESIDE
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP);E.setFromOptions()
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
    E.setTolerances(1e-06,max_it=250)
    if guess != None:
        # guess = 0.21+0.09*1j #(?)
        sinv.setType('sinvert')
        sinv.setShift(guess)

    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    if nconv <= nEV: evals = nconv
    else: evals = nEV

    for i in range(evals):
        eigVal = E.getEigenvalue(i)
        grow[i,cnt] = eigVal.imag*kx
        freq[i,cnt] = eigVal.real*kx

        # plt.plot(np.arange(0,nk), grow[i,:]*3600*24, 'o')

        E.getEigenvector(i,vr,vi)

        start,end = vi.getOwnershipRange()
        if start == 0: mode[0,i,cnt] = 0; start+=1
        if end == Ny: mode[Ny,i,cnt] = 0; end -=1

        for j in range(start,end):
            mode[j,i,cnt] = 1j*vi[j]; mode[j,i,cnt] = vr[j]
    t1 = time.time()
    print "Time for solve: ",t1-t0

    return grow,freq,mode,(grow[0,cnt]*1j+freq[0,cnt])/kx

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
opts = PETSc.Options()

nEV = opts.getInt('nev', 5)
domain = 'small'
jet = 'narrow'

if domain == 'large': Ly = 300e3
else: Ly = 100e03

if jet == 'wide': Lj = 7.5e3
else: Lj = 5e03

Ncb = opts.getInt('Ncb',10)  # Number of grid points for cheb
Nfd = opts.getInt('Nfd',100)  # Number of grid points for FD

f0 = 1.e-4
bet = 0
g0 = 9.81
g1 = 0.02
rho = 1024
dkk = 2e-2

yfd = np.linspace(0,Ly, Nfd+1)
hfd = yfd[1] - yfd[0]
e = np.ones(Nfd+1)

# FD
Dfd = sp.spdiags([-1*e, 0*e, e]/(2*hfd), [-1, 0, 1], Nfd+1,Nfd+1)
Dfd = sp.csr_matrix(Dfd)
Dfd[0, 0:3] = [-3, 4, -1]/(2*hfd)
Dfd[Nfd, Nfd-2:Nfd+1] = [1, -4, 3]/(2*hfd)

Dfd2 = sp.spdiags([e, -2*e, e]/hfd**2, [-1, 0, 1], Nfd+1, Nfd+1)
Dfd2 = sp.csr_matrix(Dfd2)
Dfd2[0, 0:3] = [1, -2, 1]/hfd**2
Dfd2[Nfd, Nfd-2:Nfd+1] = [1,-2,1]/hfd**2

# Cheb
Dcb,ycb = cheb(Ncb)
ycb     = Ly*(ycb+1)*0.5
Dcb     = 2*Dcb/Ly
Dcb2    = np.dot(Dcb,Dcb)

## Define velocity and height fields

if domain == 'large':
    edge = 150
    edgestr = 150
    thickness = 150
else:
    edge = -3.091730462756459e-07
    edgestr = 0
    thickness = 150

depth = thickness + edge
scale = (depth-edge)/2
shift = edge+scale

Etafd = -scale*np.tanh((-yfd+Ly/2)/Lj)-shift
Hfd = -Etafd
Ufd = g1/f0*Dfd*Etafd
# Ufd = np.transpose(Ufd)

Etacb = -scale*np.tanh((-ycb+Ly/2)/Lj)-shift
Hcb = -Etacb
Ucb = g1/f0*np.dot(Dcb,Etacb)

dUfd = Dfd*Ufd
dHfd = np.ravel(Dfd*Hfd)
H1fd = sp.spdiags(Hfd, 0, Nfd+1,Nfd+1)
H1fd = sp.csr_matrix(H1fd)
HDyfd = H1fd*Dfd

dUcb = np.dot(Dcb,Ucb)
dHcb = np.dot(Dcb,Hcb)
H1cb = sp.spdiags(np.ravel(Hcb), 0, Ncb+1,Ncb+1).todense()
HDycb = np.dot(H1cb,Dcb)

kk = np.arange(dkk,2+dkk,dkk)/Lj
nk = len(kk)

growcb = np.zeros([nEV,len(kk)])
freqcb = np.zeros([nEV,len(kk)])
modecb = np.zeros([3*Ncb+1,nEV,len(kk)],dtype=np.complex128)
rcb = range(1,Ncb)

growfd = np.zeros([nEV,len(kk)])
freqfd = np.zeros([nEV,len(kk)])
modefd = np.zeros([3*Nfd+1,nEV,len(kk)],dtype=np.complex128)
rfd = range(1,Nfd)

Acb = build_A_start(Ncb, Ucb, g1, dUcb, f0, Hcb, HDycb, dHcb)
Afd = build_A_start(Nfd, Ufd, g1, dUfd, f0, Hfd, HDyfd, dHfd)

cnt = 0
guess = 0.21+0.09*1j

for kx in kk[0:nk]:
    k2 = kx**2

    Acb = build_A(Acb,Ncb,g1,f0,k2,Dcb)
    Afd = build_A(Afd,Nfd,g1,f0,k2,Dfd)

    growcb,freqcb,modecb,sig0 = solve_eigensystem(Acb,kx,Ncb,nEV,growcb,freqcb,modecb,None)
    growfd,freqfd,modefd,temp = solve_eigensystem(Afd,kx,Nfd,nEV,growfd,freqfd,modefd,sig0)
    print sig0

    Print(cnt, 'kk = ', kk[cnt])
    Print('SW: One-layer Test Case, cheb')
    Print('  max growth ', growcb[0,cnt])
    Print('  frequency  ', freqcb[0,cnt])
    Print('SW: One-layer Test Case, FD')
    Print('  max growth ', growfd[0,cnt])
    Print('  frequency  ', freqfd[0,cnt])

    cnt = cnt+1

if rank == 0:
    # Plotting
    plt.rcParams["axes.titlesize"] = 10
    for i in range(nEV):
        plt.plot(kk*Lj, growcb[i,:]*3600*24,'o')
    plt.xlabel(["domain = [0, ",Ly/1e3," km]"])
    plt.ylabel('1/day')
    title = "1L SW - Tanh, cheb,  N=%d, Layer depth=%dm, Edge Depth=%dm, Jet Width= %.3ekm" % (Ncb,depth,edgestr,Lj/1e3*4)
    plt.title(title)
    plt.show()
    for i in range(nEV):
        plt.plot(kk*Lj, growfd[i,:]*3600*24,'o')
    plt.xlabel(["domain = [0, ",Ly/1e3," km]"])
    plt.ylabel('1/day')
    title = "1L SW - Tanh, FD,  N=%d, Layer depth=%dm, Edge Depth=%dm, Jet Width= %.3ekm" % (Nfd,depth,edgestr,Lj/1e3*4)
    plt.title(title)
    plt.show()


