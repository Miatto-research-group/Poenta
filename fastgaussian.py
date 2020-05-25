import numpy as np
from numba import jit

# Equa (51) in 2004.11002
# D(gamma)R(phi)B(theta1,psi1)S(z)B(theta,psi)
# D(gamma)U(W)S(z)U(V)

@jit(nopython=True)
def UW_(phi,theta1,psi1):
    """
    Arguments:
        phi (float np.array): phase rotation parameter
        theta1(float): transmissivity angle of the beamsplitter1
        psi1(float): reflection phase of the beamsplitter1
    """
    return np.array([[np.exp(1j*phi[0])*np.cos(theta1) , -np.exp(1j*phi[0])*np.exp(-1j*psi1)*np.sin(theta1)],[np.exp(1j*phi[1])*np.exp(1j*psi1)*np.sin(theta1), np.exp(1j*phi[1])*np.cos(theta1)]])
    
    
    
@jit(nopython=True)
def UV_(theta,psi):
    """
    Arguments:
        theta(float): transmissivity angle of the beamsplitter1
        psi(float): reflection phase of the beamsplitter1
    """
    return np.array([[np.cos(theta) , -np.exp(-1j*psi)*np.sin(theta)],[np.exp(1j*psi)*np.sin(theta), np.cos(theta)]])




@jit(nopython=True)
def C_(gamma, W, zeta, V):
    """
    C constant (0000 element of the two-mode Gaussian transformation matrix)

    Arguments:
        gamma (complex np.array): displacement parameter
        W(complex np.array): general multimode passive transformation (rotation+BS)
        zeta (complex np.array): squeezing parameter
        V(complex np.array): general multimode passive transformation (BS)
    """
    r = np.abs(zeta)
    delta = np.angle(zeta)

    return np.exp(-0.5*np.sum(np.abs(gamma)**2) + np.dot(np.dot(np.conj(gamma).T,(W*np.diag(np.exp(1j*delta)*np.tanh(r))*W.T)),np.conj(gamma)))/np.sum(np.sqrt(np.cosh(r)))

@jit(nopython=True)
def mu_(gamma, W, zeta, V):
    """
    Mu vector

    Arguments:
        gamma (complex np.array): displacement parameter
        W(complex np.array): general multimode passive transformation (rotation+BS)
        zeta (complex np.array): squeezing parameter
        V(complex np.array): general multimode passive transformation (BS)
    """
    r = np.abs(zeta)
    delta = np.angle(zeta)
    mu1 = np.dot(np.conj(gamma).T,(W*np.diag(np.exp(1j*delta)*np.tanh(r))*W.T)) 
    mu2 = -np.dot(np.conj(gamma).T,(W*np.diag(1/np.cosh(r))*V)) 
    return np.concatenate([mu1, mu2])


@jit(nopython=True)
def Sigma_(gamma, W, zeta, V):
    """
    Sigma matrix

    Arguments:
        gamma (complex np.array): displacement parameter
        W(complex np.array): general multimode passive transformation (rotation+BS)
        zeta (complex np.array): squeezing parameter
        V(complex np.array): general multimode passive transformation (BS)
    """
    r = np.abs(zeta)
    delta = np.angle(zeta)
    
    return np.concatenate([np.concatenate( [W*np.diag(np.exp(1j*delta)*np.tanh(r))*W.T, -W*np.diag(1/np.cosh(r))*V],axis=1),
                     np.concatenate([-V.T*np.diag(1/np.cosh(r))*W.T, -V.T*np.diag(np.exp(-1j*delta)*np.tanh(r))*V],axis=1)])

@jit(nopython=True)
def new_state(gamma, phi, theta1, psi1, zeta, theta, psi, old_state):
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        
        theta1(float): transmissivity angle of the beamsplitter1
        psi1(float): reflection phase of the beamsplitter1
        
        zeta (complex np.array): squeezing parameter
        
        theta(float): transmissivity angle of the beamsplitter
        psi(float): reflection phase of the beamsplitter
        
        old_state (np.array(complex)): State to be transformed

    Returns:
        (np.array(complex)): the transformed state
    """
    
    W = UW_(phi,theta1,psi1)
    V = UV_(theta,psi)
    
    C = C_(gamma, W, zeta, V)
    mu = mu_(gamma, W, zeta, V)
    Sigma = Sigma_(gamma, W, zeta, V)

    cutoff = old_state.shape[0]
    dtype = old_state.dtype

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    sqrts = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            sqrts[m,n]= np.sqrt(m+1)*np.sqrt(n+1)


    R = np.zeros((cutoff, cutoff, cutoff+1, cutoff+1), dtype=dtype)
    G_mn00 = np.zeros((cutoff, cutoff), dtype=dtype)
    
    
    #G_mn00
    G_mn00[0,0] = C
    for n in range(1, cutoff):
        G_mn00[0,n] = mu[3]/sqrt[n]*G_mn00[0,n-1] - Sigma[3,3]*sqrt[n-1]/sqrt[n]*G_mn00[0,n-2]


    for m in range(1,cutoff):
        for n in range(cutoff):
            G_mn00[m,n] = mu[2]/sqrt[m]*G_mn00[m-1,n] - Sigma[2,2]*sqrt[m-1]/sqrt[m]*G_mn00[m-2,n] - Sigma[2,3]*sqrt[n]/sqrt[m]*G_mn00[m-1,n-1]
            
    # R_00^jk = G_mn00 * a^j b^k|old_state>
    for j in range(cutoff):
        for k in range(cutoff):
            R[0,0,j,k] = np.sum(G_mn00[:cutoff - j,:cutoff - k]*(old_state[j:, k:]*sqrts[ : cutoff-j,:cutoff-k]))
            
    #R_0n^jk
    for n in range(1,cutoff):
        for j in range(cutoff):
            for k in range(cutoff):
                R[0,n,j,k] = mu[1]/sqrt[n]*R[0,n-1,j,k] - Sigma[1,1]/sqrt[n]*sqrt[n-1]*R[0,n-2,j,k] - Sigma[1,2]/sqrt[n]*R[0,n-1,j+1,k] - Sigma[1,3]/sqrt[n]*R[0,m-1,j,k+1]
                
    #R_mn^jk
    for m in range(1,cutoff):
        for n in range(cutoff):
            for j in range(cutoff - m):
                for k in range(cutoff-m-j):
                    R[m,n,j,k] = mu[0]/sqrt[m]*R[m-1,n,j,k] - Sigma[0,0]/sqrt[m]*sqrt[m-1]*R[m-2,n,j,k] - Sigma[0,1]*sqrt[n]/sqrt[m]*R[m-1,n-1,j,k] - Sigma[0,2]/sqrt[m]*R[m-1,n,j+1,k] - Sigma[0,3]/sqrt[m]*R[m-1,n,j,k+1]
                    
    return R[:,:,0,0]