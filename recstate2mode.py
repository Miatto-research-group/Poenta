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
    return np.array([[np.exp(1j*phi[0])*np.cos(theta1) , -np.exp(1j*phi[0])*np.exp(-1j*psi1)*np.sin(theta1)],[np.exp(1j*phi[1])*np.exp(1j*psi1)*np.sin(theta1), np.exp(1j*phi[1])*np.cos(theta1)]],dtype=np.complex128)
    
    
    
@jit(nopython=True)
def UV_(theta,psi):
    """
    Arguments:
        theta(float): transmissivity angle of the beamsplitter1
        psi(float): reflection phase of the beamsplitter1
    """
    return np.array([[np.cos(theta) , -np.exp(-1j*psi)*np.sin(theta)],[np.exp(1j*psi)*np.sin(theta), np.cos(theta)]],dtype=np.complex128)




@jit(nopython=True)
def C_(gamma, phi, theta1, psi1, zeta, theta, psi):
    """
    C constant (0000 element of the two-mode Gaussian transformation matrix)

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        theta1(float): transmissivity angle of the beamsplitter1
        psi1(float): reflection phase of the beamsplitter1
        zeta (complex np.array): squeezing parameter
        theta(float): transmissivity angle of the beamsplitter
        psi(float): reflection phase of the beamsplitter
    """
    r = np.abs(zeta)
    delta = np.angle(zeta)

    return np.exp(-0.5*np.sum(np.abs(gamma)**2) - np.dot(np.dot(np.conj(gamma).T,(W*np.diag(np.exp(1j*delta)*np.tanh(r))*W.T)),np.conj(gamma))) / np.sqrt(np.cosh(r)[0]*np.cosh(r)[1])

@jit(nopython=True)
def mu_(gamma, phi, theta1, psi1, zeta, theta, psi):
    """
    Mu vector

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        theta1(float): transmissivity angle of the beamsplitter1
        psi1(float): reflection phase of the beamsplitter1
        zeta (complex np.array): squeezing parameter
        theta(float): transmissivity angle of the beamsplitter
        psi(float): reflection phase of the beamsplitter
    """
    W = UW_(phi,theta1,psi1)
    V = UV_(theta,psi)
    r = np.abs(zeta)
    delta = np.angle(zeta)
    mu = np.zeros(4,dtype = np.complex128)
    mu[:2] = np.dot(np.conj(gamma).T,(W*np.diag(np.exp(1j*delta)*np.tanh(r))*W.T)) + np.conj(gamma)
    mu[2:] = -np.dot(np.conj(gamma).T,(W*np.diag(1/np.cosh(r))*V)) 
    return mu


@jit(nopython=True)
def Sigma_(gamma, phi, theta1, psi1, zeta, theta, psi):
    """
    Sigma matrix

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        theta1(float): transmissivity angle of the beamsplitter1
        psi1(float): reflection phase of the beamsplitter1
        zeta (complex np.array): squeezing parameter
        theta(float): transmissivity angle of the beamsplitter
        psi(float): reflection phase of the beamsplitter
    """
    W = UW_(phi,theta1,psi1)
    V = UV_(theta,psi)
    r1,r2 = np.abs(zeta)
    delta1,delta2 = np.angle(zeta)
    
    diagtanh = np.array([[np.exp(1j*delta1)*np.tanh(r1),0],[0,np.exp(1j*delta2)*np.tanh(r2)]],dtype=np.complex128)
    diagtanh_minus = np.array([[np.exp(-1j*delta1)*np.tanh(r1),0],[0,np.exp(-1j*delta2)*np.tanh(r2)]],dtype=np.complex128)
    diagsech = np.array([[1/np.cosh(r1),0],[0,1/np.cosh(r2)]],dtype=np.complex128)
    
    W1 = W@(diagtanh@W.T)
    W2 = -W@(diagsech@V)
    W3 = -V.T@(diagsech@W.T)
    W4 = -V.T@(diagtanh_minus@V)

    
    return np.concatenate((np.concatenate( (W1,W2) ,axis=1),
                     np.concatenate((W3, W4),axis=1)))

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
    Sigma = Sigma_(W, zeta, V)

    cutoff = old_state.shape[0]
    dtype = old_state.dtype

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    sqrts = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            sqrts[m,n]= np.sqrt(m+1)*np.sqrt(n+1)


    R = np.zeros((cutoff, cutoff, cutoff+1, cutoff+1), dtype=dtype)
    G_00pq = np.zeros((cutoff, cutoff), dtype=dtype)
    
    
    #G_mn00
    G_00pq[0,0] = C
    for q in range(1, cutoff):
        G_00pq[0,q] = (mu[3]*G_00pq[0,q-1] - Sigma[3,3]*sqrt[q-1]*G_00pq[0,q-2])/sqrt[q]


    for p in range(1,cutoff):
        for q in range(0,cutoff):
            G_00pq[p,q] = (mu[2]*G_00pq[p-1,q] - Sigma[2,2]*sqrt[p-1]*G_00pq[p-2,q] - Sigma[2,3]*sqrt[q]*G_00pq[p-1,q-1])/sqrt[p]
                    
    # R_00^jk = a_dagger^j \G_00pq> b^k  * |old_state>
    a = np.zeros((cutoff,cutoff), dtype=dtype)
    for i in range(cutoff-1):
        a[i+1,i] = np.sqrt(i+1)
    b = np.zeros((cutoff,cutoff), dtype=dtype)
    for i in range(cutoff-1):
        b[i,i+1] = np.sqrt(i+1)


    for j in range(cutoff):
        for k in range(cutoff):
            G_00pq2 = G_00pq
            for _ in range(j):
                G_00pq2 = a@G_00pq2
            for _ in range(k):
                G_00pq2 = G_00pq2@b
            R[0,0,j,k] = np.sum(G_00pq2*old_state)

    #R_0n^jk
    for n in range(1,cutoff):
        for k in range(0,cutoff): 
            for j in range(0,cutoff):
                R[0,n,j,k] = mu[1]/sqrt[n]*R[0,n-1,j,k] - Sigma[1,1]/sqrt[n]*sqrt[n-1]*R[0,n-2,j,k] - Sigma[1,2]/sqrt[n]*R[0,n-1,j+1,k] - Sigma[1,3]/sqrt[n]*R[0,n-1,j,k+1]


    for m in range(1,cutoff):
        for n in range(0,cutoff):
            for j in range(0,cutoff-m):
                for k in range(0,cutoff-m-j):
                    R[m,n,j,k] = mu[0]/sqrt[m]*R[m-1,n,j,k] - Sigma[0,0]/sqrt[m]*sqrt[m-1]*R[m-2,n,j,k] - Sigma[0,1]*sqrt[n]/sqrt[m]*R[m-1,n-1,j,k] - Sigma[0,2]/sqrt[m]*R[m-1,n,j+1,k] - Sigma[0,3]/sqrt[m]*R[m-1,n,j,k+1]


                    
    return R[:,:,0,0]

@jit(nopython=True)
def RG_00pq(gamma, phi, theta1, psi1, zeta, theta, psi, old_state):
    """
    Get the R matrix and G_00pq matrix

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
        R, G_00pq (np.array(complex), np.array(complex)): The R and G_00pq matrices
    """
    
    W = UW_(phi,theta1,psi1)
    V = UV_(theta,psi)
    
    C = C_(gamma, W, zeta, V)
    mu = mu_(gamma, W, zeta, V)
    Sigma = Sigma_(W, zeta, V)

    cutoff = old_state.shape[0]
    dtype = old_state.dtype

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))

    R = np.zeros((cutoff, cutoff, cutoff+1, cutoff+1), dtype=dtype)
    G_00pq = np.zeros((cutoff, cutoff), dtype=dtype)
    
    G_00pq[0,0] = C
    for q in range(1, cutoff):
        G_00pq[0,q] = (mu[3]*G_00pq[0,q-1] - Sigma[3,3]*sqrt[q-1]*G_00pq[0,q-2])/sqrt[q]


    for p in range(1,cutoff):
        for q in range(0,cutoff):
            G_00pq[p,q] = (mu[2]*G_00pq[p-1,q] - Sigma[2,2]*sqrt[p-1]*G_00pq[p-2,q] - Sigma[2,3]*sqrt[q]*G_00pq[p-1,q-1])/sqrt[p]
                    
    # R_00^jk = a_dagger^j \G_00pq> b^k  * |old_state>
    a = np.zeros((cutoff,cutoff), dtype=dtype)
    for i in range(cutoff-1):
        a[i+1,i] = np.sqrt(i+1)
    b = np.zeros((cutoff,cutoff), dtype=dtype)
    for i in range(cutoff-1):
        b[i,i+1] = np.sqrt(i+1)


    for j in range(cutoff):
        for k in range(cutoff):
            G_00pq2 = G_00pq
            for _ in range(j):
                G_00pq2 = a@G_00pq2
            for _ in range(k):
                G_00pq2 = G_00pq2@b
            R[0,0,j,k] = np.sum(G_00pq2*old_state)

    #R_0n^jk
    for n in range(1,cutoff):
        for k in range(0,cutoff): 
            for j in range(0,cutoff):
                R[0,n,j,k] = mu[1]/sqrt[n]*R[0,n-1,j,k] - Sigma[1,1]/sqrt[n]*sqrt[n-1]*R[0,n-2,j,k] - Sigma[1,2]/sqrt[n]*R[0,n-1,j+1,k] - Sigma[1,3]/sqrt[n]*R[0,n-1,j,k+1]


    for m in range(1,cutoff):
        for n in range(0,cutoff):
            for j in range(0,cutoff-m):
                for k in range(0,cutoff-m-j):
                    R[m,n,j,k] = mu[0]/sqrt[m]*R[m-1,n,j,k] - Sigma[0,0]/sqrt[m]*sqrt[m-1]*R[m-2,n,j,k] - Sigma[0,1]*sqrt[n]/sqrt[m]*R[m-1,n-1,j,k] - Sigma[0,2]/sqrt[m]*R[m-1,n,j+1,k] - Sigma[0,3]/sqrt[m]*R[m-1,n,j,k+1]


                    
    return R,G_00pq

@jit(nopython = True)
def grad_state(gamma, phi, theta1, psi1, zeta, theta, psi, old_state):
    """
    Gradient of the new state
    """
    
    
    C = C_(gamma, phi, theta1, psi1, zeta, theta, psi)
    mu = mu_(gamma, phi, theta1, psi1, zeta, theta, psi)
    Sigma = Sigma_(gamma, phi, theta1, psi1, zeta, theta, psi)
    dC = dC_(gamma, phi, theta1, psi1, zeta, theta, psi)
    dmu = dmu_(gamma, phi, theta1, psi1, zeta, theta, psi)
    dSigma = dSigma_(gamma, phi, theta1, psi1, zeta, theta, psi)
    
    R, G_00pq = RG_00pq(gamma, phi, theta1, psi1, zeta, theta, psi, old_state)
    
    cutoff = len(old_state)
    dtype = old_state.dtype
    
    dR = np.zeros((cutoff, cutoff, cutoff , cutoff, 14), dtype=dtype)
    dG_00pq = np.zeros((cutoff, cutoff, 14), dtype=dtype)
    
    dG_00pq[0,0] = dC
    for q in range(1, cutoff):
        dG_00pq[0,q] = (dmu[3]*G_00pq[0,q-1]+mu[3]*dG_00pq[0,q-1] - dSigma[3,3]*sqrt[q-1]*G_00pq[0,q-2]- Sigma[3,3]*sqrt[q-1]*dG_00pq[0,q-2])/sqrt[q]


    for p in range(1,cutoff):
        for q in range(0,cutoff):
            dG_00pq[p,q] = (dmu[2]*G_00pq[p-1,q]+ mu[2]*dG_00pq[p-1,q] - dSigma[2,2]*sqrt[p-1]*G_00pq[p-2,q]- Sigma[2,2]*sqrt[p-1]*dG_00pq[p-2,q] - dSigma[2,3]*sqrt[q]*G_00pq[p-1,q-1]- Sigma[2,3]*sqrt[q]*dG_00pq[p-1,q-1])/sqrt[p]

    
    # R_00^jk = a_dagger^j \G_00pq> b^k  * |old_state>
    a = np.zeros((cutoff,cutoff), dtype=dtype)
    for i in range(cutoff-1):
        a[i+1,i] = np.sqrt(i+1)
    b = np.zeros((cutoff,cutoff), dtype=dtype)
    for i in range(cutoff-1):
        b[i,i+1] = np.sqrt(i+1)


    for j in range(cutoff):
        for k in range(cutoff):
            dG_00pq2 = dG_00pq
            for _ in range(j):
                dG_00pq2 = a@dG_00pq2
            for _ in range(k):
                dG_00pq2 = dG_00pq2@b
            dR[0,0,j,k] = np.sum(dG_00pq2*old_state)

    for n in range(1,cutoff):
        for k in range(0,cutoff): 
            for j in range(0,cutoff):
                dR[0,n,j,k] = dmu[1]/sqrt[n]*R[0,n-1,j,k] + mu[1]/sqrt[n]*dR[0,n-1,j,k] - dSigma[1,1]/sqrt[n]*sqrt[n-1]*R[0,n-2,j,k] - Sigma[1,1]/sqrt[n]*sqrt[n-1]*dR[0,n-2,j,k] - dSigma[1,2]/sqrt[n]*R[0,n-1,j+1,k] - Sigma[1,2]/sqrt[n]*dR[0,n-1,j+1,k] - dSigma[1,3]/sqrt[n]*R[0,n-1,j,k+1] - Sigma[1,3]/sqrt[n]*dR[0,n-1,j,k+1]


    for m in range(1,cutoff):
        for n in range(0,cutoff):
            for j in range(0,cutoff-m):
                for k in range(0,cutoff-m-j):
                    dR[m,n,j,k] = dmu[0]/sqrt[m]*R[m-1,n,j,k] + mu[0]/sqrt[m]*dR[m-1,n,j,k] - dSigma[0,0]/sqrt[m]*sqrt[m-1]*R[m-2,n,j,k] - Sigma[0,0]/sqrt[m]*sqrt[m-1]*dR[m-2,n,j,k] - dSigma[0,1]*sqrt[n]/sqrt[m]*R[m-1,n-1,j,k] - Sigma[0,1]*sqrt[n]/sqrt[m]*dR[m-1,n-1,j,k] - dSigma[0,2]/sqrt[m]*R[m-1,n,j+1,k] - Sigma[0,2]/sqrt[m]*dR[m-1,n,j+1,k] - dSigma[0,3]/sqrt[m]*R[m-1,n,j,k+1] - Sigma[0,3]/sqrt[m]*dR[m-1,n,j,k+1]
           
    return np.transpose(dR[:,:,0,0])


@jit(nopython=True)
def G_matrix(gamma, phi, theta1, psi1, zeta, theta, psi,cutoff):
    """
    Directly constructs the transformation G matrix recursively and exactly.

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        
        theta1(float): transmissivity angle of the beamsplitter1
        psi1(float): reflection phase of the beamsplitter1
        
        zeta (complex np.array): squeezing parameter
        
        theta(float): transmissivity angle of the beamsplitter
        psi(float): reflection phase of the beamsplitter

    Returns:
        (np.array(complex)): the transformation matrix G
    """
    
    W = UW_(phi,theta1,psi1)
    V = UV_(theta,psi)
    
    C = C_(gamma, W, zeta, V)
    mu = mu_(gamma, W, zeta, V)
    Sigma = Sigma_( W, zeta, V)


    sqrt = np.sqrt(np.arange(cutoff))


    G = np.zeros((cutoff, cutoff, cutoff, cutoff),dtype=np.complex128)
    
    

    G[0,0,0,0] = C
    

    for q in range(1, cutoff):
        G[0,0,0,q] = mu[3]/sqrt[q]*G[0,0,0,q-1] - Sigma[3,3]*sqrt[q-1]/sqrt[q]*G[0,0,0,q-2]


    for q in range(0,cutoff):
        for p in range(1,cutoff):
            G[0,0,p,q] = mu[2]/sqrt[p]*G[0,0,p-1,q] - Sigma[2,2]*sqrt[p-1]/sqrt[p]*G[0,0,p-2,q] - Sigma[2,3]*sqrt[q]/sqrt[p]*G[0,0,p-1,q-1]

    for q in range(0,cutoff):
        for p in range(0,cutoff):
            for n in range(1,cutoff):
                G[0,n,p,q] = mu[1]/sqrt[n]*G[0,n-1,p,q] - Sigma[1,1]/sqrt[n]*sqrt[n-1]*G[0,n-2,p,q] - Sigma[1,2]/sqrt[n]*sqrt[p]*G[0,n-1,p-1,q] - Sigma[1,3]/sqrt[n]*sqrt[q]*G[0,n-1,p,q-1]
                

    for q in range(0,cutoff):
        for p in range(0,cutoff):
            for n in range(0,cutoff):
                for m in range(1,cutoff):
                    G[m,n,p,q] = mu[0]/sqrt[m]*G[m-1,n,p,q] - Sigma[0,0]/sqrt[m]*sqrt[m-1]*G[m-2,n,p,q] - Sigma[0,1]*sqrt[n]/sqrt[m]*G[m-1,n-1,p,q] - Sigma[0,2]*sqrt[p]/sqrt[m]*G[m-1,n,p-1,q] - Sigma[0,3]*sqrt[q]/sqrt[m]*G[m-1,n,p,q-1]
                    
    return G

@jit(nopython=True)
def G_matrix2(gamma, phi, theta1, psi1, zeta, theta, psi,cutoff):
    """
    Directly constructs the transformation G matrix recursively and exactly.
    #Using a different index order with G_matrix.#

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        
        theta1(float): transmissivity angle of the beamsplitter1
        psi1(float): reflection phase of the beamsplitter1
        
        zeta (complex np.array): squeezing parameter
        
        theta(float): transmissivity angle of the beamsplitter
        psi(float): reflection phase of the beamsplitter

    Returns:
        (np.array(complex)): the transformation matrix G
    """
    
    W = UW_(phi,theta1,psi1)
    V = UV_(theta,psi)
    
    C = C_(gamma, W, zeta, V)
    mu = mu_(gamma, W, zeta, V)
    Sigma = Sigma_( W, zeta, V)


    sqrt = np.sqrt(np.arange(cutoff))


    G = np.zeros((cutoff, cutoff, cutoff, cutoff),dtype=np.complex128)
    
    
    #G_0000
    G[0,0,0,0] = C
    
    #G_000q
    for m in range(1, cutoff):
        G[m,0,0,0] = (mu[0]*G[m-1,0,0,0] - Sigma[0,0]*sqrt[m-1]*G[m-2,0,0,0] )/sqrt[m]


    for m in range(0,cutoff):
        for n in range(1,cutoff):        
            G[m,n,0,0] = (mu[1]*G[m,n-1,0,0] - Sigma[1,0]*sqrt[m]*G[m-1,n-1,0,0] - Sigma[1,1]*sqrt[n-1]*G[m,n-2,0,0])/sqrt[n]

    for m in range(0,cutoff):
        for n in range(0,cutoff):
            for p in range(1,cutoff):
                G[m,n,p,0] = (mu[2]*G[m,n,p-1,0] - Sigma[2,0]*sqrt[m]*G[m-1,n,p-1,0] - Sigma[2,1]*sqrt[n-1]*G[m,n-1,p-1,0] - Sigma[2,2]*sqrt[p-1]*G[m,n,p-2,0] )/sqrt[p]
                
                
    #G_mn^jk
    for m in range(0,cutoff):
        for n in range(0,cutoff):
            for p in range(0,cutoff):
                for q in range(1,cutoff):
                    G[m,n,p,q] =( mu[3]*G[m,n,p,q-1] - Sigma[3,0]*sqrt[m]*G[m-1,n,p,q-1] - Sigma[3,1]*sqrt[n]*G[m,n-1,p,q-1] - Sigma[3,2]*sqrt[p]*G[m,n,p-1,q-1] - Sigma[3,3]*sqrt[q-1]*G[m,n,p,q-2] )/sqrt[q]
                    
    return G





