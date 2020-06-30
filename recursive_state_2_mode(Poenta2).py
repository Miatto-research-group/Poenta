import numpy as np
from numba import jit

@jit(nopython=True)
def C_mu_Sigma(gamma, phi, theta1, varphi1, zeta, theta, varphi, dtype = np.complex128):
    """
    Utility function to construct:
    1. C constant
    2. Mu vector
    3. Sigma matrix

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        theta1(float): transmissivity angle of the beamsplitter1
        varphi1(float): reflection phase of the beamsplitter1
        zeta (complex np.array): squeezing parameter
        theta(float): transmissivity angle of the beamsplitter
        varphi(float): reflection phase of the beamsplitter
        dtype (numpy type): unused for now

    Returns:
        C (complex), mu (complex array[4]), Sigma (complex array[4,4])
    """
    r = np.abs(zeta)
    delta = np.angle(zeta)
    W = np.array([[np.exp(1j*phi[0])*np.cos(theta1) , -np.exp(1j*phi[0])*np.exp(-1j*varphi1)*np.sin(theta1)],[np.exp(1j*phi[1])*np.exp(1j*varphi1)*np.sin(theta1), np.exp(1j*phi[1])*np.cos(theta1)]],dtype=dtype)
    V = np.array([[np.cos(theta) , -np.exp(-1j*varphi)*np.sin(theta)],[np.exp(1j*varphi)*np.sin(theta), np.cos(theta)]],dtype=dtype)

    C = np.exp(-0.5*np.sum(np.abs(gamma)**2) - np.dot(np.dot(np.conj(gamma).T,(W*np.diag(np.exp(1j*delta)*np.tanh(r))*W.T)),np.conj(gamma))) / np.sqrt(np.cosh(r)[0]*np.cosh(r)[1])
    
    mu = np.zeros(4,dtype = dtype)
    mu[:2] = np.dot(np.conj(gamma).T,(W*np.diag(np.exp(1j*delta)*np.tanh(r))*W.T)) + np.conj(gamma)
    mu[2:] = -np.dot(np.conj(gamma).T,(W*np.diag(1/np.cosh(r))*V)) 
    
    r1,r2 = np.abs(zeta)
    delta1,delta2 = np.angle(zeta)
    diagtanh = np.array([[np.exp(1j*delta1)*np.tanh(r1),0],[0,np.exp(1j*delta2)*np.tanh(r2)]],dtype=dtype)
    diagtanh_minus = np.array([[np.exp(-1j*delta1)*np.tanh(r1),0],[0,np.exp(-1j*delta2)*np.tanh(r2)]],dtype=dtype)
    diagsech = np.array([[1/np.cosh(r1),0],[0,1/np.cosh(r2)]],dtype=dtype)
    W1 = W@(diagtanh@W.T)
    W2 = -W@(diagsech@V)
    W3 = -V.T@(diagsech@W.T)
    W4 = -V.T@(diagtanh_minus@V)
    Sigma = np.concatenate((np.concatenate( (W1,W2) ,axis=1),np.concatenate((W3, W4),axis=1))) 
    return C, mu, Sigma

@jit(nopython=True)
def dC_dmu_dSigma(gamma, phi, theta1, varphi1, zeta, theta, varphi, dtype = np.complex128):
    """
    Utility function to construct the gradient of:
    1. C constant
    2. Mu vector
    3. Sigma matrix
    with respect to gamma1, gamma1*, gamma2, gamma2*, phi1, phi2, theta1, varphi1, zeta1, zeta1*, zeta2, zeta2*, theta, varphi

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        theta1(float): transmissivity angle of the beamsplitter1
        varphi1(float): reflection phase of the beamsplitter1
        zeta (complex np.array): squeezing parameter
        theta(float): transmissivity angle of the beamsplitter
        varphi(float): reflection phase of the beamsplitter
        dtype (numpy type): unused for now

    Returns:
        dC (complex array[12]), dmu (complex array[4,12]), dSigma (complex array[4,4,12])
    """
    C, mu, Sigma = C_mu_Sigma(gamma, phi, theta1, varphi1, zeta, theta, varphi)
    
    gamma1, gamma2 = gamma
    gamma1c, gamma2c = np.conj(gammma)
    phi1, phi2 = phi
    r1, r2 = np.abs(zeta)
    delta1, delta2 = np.angle(zeta)
    e_iphi1 = np.exp(1j*phi1)
    e_iphi2 = np.exp(1j*phi2)
    e_iphi12 = np.exp(1j*(phi1+phi2))
    e_ivarphi1 = np.exp(1j*varphi1)
    e_ivarphi = np.exp(1j*varphi)
    T1 = np.exp(1j*delta1)*np.tanh(r1)
    T2 = np.exp(1j*delta2)*np.tanh(r2)
    

    # dC
    dC_dgamma1 = -C*gamma1c/2
    dC_dgamma1c = -C/2*(gamma1+2*gamma1c*(np.cos(theta1)**2*T1+e_ivarphi1**(-2)*np.sin(theta1)**2*T2) + gamma2c*e_iphi12*np.sin(2*theta1)(e_ivarphi1*T1-e_ivarphi1**(-1)*T2))
    dC_dgamma2 = -C*gamma2c/2
    dC_dgamma2c = -C/2*(gamma2+2*gamma2c*(np.cos(theta1)**2*T2+e_ivarphi1**2*np.sin(theta1)**2*T1) + gamma1c*e_iphi12*np.sin(2*theta1)(e_ivarphi1*T1-e_ivarphi1**(-1)*T2))
    dC_dphi1 = -C/2*(2j*gamma1c**2*(np.cos(theta1)**2*T1+e_ivarphi1**(-2)*np.sin(theta1)**2*T2)+1j*gamma1c*gamma2c*e_iphi12*np.sin(2*theta1)*(e_ivarphi1*T1-e_ivarphi1**(-1)*T2))
    dC_dphi2 = -C/2*(2j*gamma2c**2*(np.cos(theta1)**2*T2+e_ivarphi1**2*np.sin(theta1)**2*T2)+1j*gamma1c*gamma2c*e_iphi12*np.sin(2*theta1)*(e_ivarphi1*T1-e_ivarphi1**(-1)*T2))
    dC_dtheta1 = -C/2*(T1-e_ivarphi1**(-2)*T2)(-gamma1c**2*e_iphi1**2*np.sin(2*theta1)+2*gamma1c*gamma2c*e_iphi12*e_ivarphi1*np.cos(2*theta1)+gamma2c**2*e_iphi2**2*np.sin(2*theta1)*e_ivarphi1**2)
    dC_dvarphi1 = -1j*C*np.sin(theta1)(-gamma1c**2*e_iphi1**2*e_ivarphi1**(-2)*np.sin(theta1)*T2 + gamma1c*gamma2c*e_iphi12*np.cos(theta1)(e_ivarphi1*T1+e_ivarphi1**(-1)*T2)+gamma2c**2*e_iphi2**2*e_ivarphi1**2*np.sin(theta1)*T1 )
    
    dC_dr1 = -1/2*np.tanh(r1)*C - C/2*np.exp(1j*delta1)*np.sech(r1)**2*(gamma1c*e_iphi1*np.cos(theta1)+gamma2c*e_iphi2*e_ivarphi1*np.sin(theta1))**2
    dC_ddelta1 = -1/2*np.tanh(r1)*C - 1j*C/2*np.exp(1j*delta1)*np.tanh(r1)*(gamma1c*e_iphi1*np.cos(theta1)+gamma2c*e_iphi2*e_ivarphi1*np.sin(theta1))**2
    if r > 0.01:
        dC_ddelta1_over_r = dC_ddelta1/r
    else: # Taylor series for tanh(r)/r
        dC_ddelta1_over_r = 
    dC_dzeta1 = np.exp(-1j*delta1)*(dC_dr1 - 1j*dC_ddelta1_over_r)/2
    dC_dzeta1c = np.exp(1j*delta1)*(dC_dr1 + 1j*dC_ddelta1_over_r)/2
    
    dC_dr2 = -1/2*np.tanh(r2)*C - C/2*np.exp(1j*delta2)*np.sech(r2)**2*(gamma1c*e_iphi1*e_ivarphi1**(-1)*np.sin(theta1)-gamma2c*e_iphi2*np.sin(theta1))**2
    dC_ddelta2 = -1/2*np.tanh(r2)*C - 1j*C/2*np.exp(1j*delta2)*np.tanh(r2)*e_ivarphi1**(-2)*(-gamma1c*e_iphi1*np.sin(theta1)+gamma2c*e_iphi2*e_ivarphi1*np.cos(theta1))**2
    if r > 0.01:
        dC_ddelta2_over_r = dC_ddelta2/r
    else: # Taylor series for tanh(r)/r
        dC_ddelta2_over_r = 
    dC_dzeta2 = np.exp(-1j*delta2)*(dC_dr2 - 1j*dC_ddelta2_over_r)/2
    dC_dzeta2c = np.exp(1j*delta2)*(dC_dr2 + 1j*dC_ddelta2_over_r)/2

    dC_dtheta = 0
    dC_dvarphi = 0
    
    dC = np.array([dC_dgamma1, dC_dgamma1c, dC_dgamma2, dC_dgamma2c, dC_dphi1, dC_dphi2, dC_dtheta1, dC_dvarphi1, dC_dzeta1, dC_dzeata1c, dC_dzeta2, dC_dzeata2c, dC_dtheta, dC_dvarphi])

    # dmu
    dmu_dgamma1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    dmu_dgamma1c = np.array([e_iphi1**2*(e_ivarphi1**(-2)*T2*np.sin(theta1)**2+T1*np.cos(theta1)**2), e_iphi12*np.sin(theta1)*np.cos(theta1)*(e_ivarphi1*T1-e_ivarphi1**(-1)*T2),-e_iphi1*np.cos(theta)*np.cos(theta1)*np.sech(r1)+e_iphi1*e_ivarphi*e_ivarphi1**(-1)*np.sin(theta)*np.sin(theta1)*np.sech(r2) , e_iphi1*e_ivarphi**(-1)*np.cos(theta)*np.sin(theta1)*np.sech(r1)+e_iphi1*e_ivarphi1**(-1)*np.cos(theta)*np.sin(theta1)*np.sech(r2)], dtype=dtype)
    dmu_dgamma2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=dtype)
    dmu_dgamma2c = np.array([e_iphi12*np.sin(theta1)*np.cos(theta1)*(e_ivarphi1*T1-e_ivarphi1**(-1)*T2), e_iphi2**2(T2*np.cos(theta1)**2+e_ivarphi1**2*T1*np.sin(theta1)**2),-e_iphi2(e_ivarphi*np.cos(theta)*np.cos(theta1)*np.sech(r2) + e_ivarphi1*np.sin(theta1)*np.cos(theta)*np.sech(r1)) , e_iphi2(-np.cos(theta)*np.cos(theta1)*np.sech(r2) + e_ivarphi**(-1)*e_ivarphi1*np.sin(theta1)*np.sin(theta)*np.sech(r1))], dtype=dtype)
    
    dmu_dphi1 = np.array([ , , , ], dtype=dtype)
    dmu_dphi2 = np.array([ , , , ], dtype=dtype)
    
    dmu_dtheta1 = np.array([ , , , ], dtype=dtype)
    dmu_dvarphi1 = np.array([ , , , ], dtype=dtype)
    
    dmu_dr1 = np.array([ , , , ], dtype=dtype)
    dmu_ddelta1 = np.array([ , , , ], dtype=dtype)
    if r > 0.01:
        dmu_ddelta1_over_r = dmu_ddelta/r
    else: # Taylor series for tanh(r)/r
        dmu_ddelta1_over_r = 
    dmu_dzeta1 = np.exp(-1j*delta)*(dmu_dr1 - 1j*dmu_ddelta1_over_r)/2
    dmu_dzeta1c = np.exp(1j*delta)*(dmu_dr1 + 1j*dmu_ddelta1_over_r)/2
    
    dmu_dr2 = np.array([ , , , ], dtype=dtype)
    dmu_ddelta2 = np.array([ , , , ], dtype=dtype)
    if r > 0.01:
        dmu_ddelta2_over_r = dmu_ddelta/r
    else: # Taylor series for tanh(r)/r
        dmu_ddelta2_over_r = 
    dmu_dzeta2 = np.exp(-1j*delta)*(dmu_dr2 - 1j*dmu_ddelta2_over_r)/2
    dmu_dzeta2c = np.exp(1j*delta)*(dmu_dr2 + 1j*dmu_ddelta2_over_r)/2
    
    
    dmu_dtheta = np.array([0.0, 0.0, , ], dtype=dtype)
    dmu_dvarphi = np.array([0.0, 0.0, , ], dtype=dtype)
    
    
    dmu = np.stack((dmu_dgamma1, dmu_dgamma1c, dmu_dgamma2, dmu_dgamma2c, dmu_dphi1, dmu_dphi2, dmu_dtheta1, dmu_dvarphi1, dmu_dzeta1, dmu_dzeata1c, dmu_dzeta2, dmu_dzeata2c, dmu_dtheta, dmu_dvarphi), axis=-1)

    # dSigma
#     dSigma_dgamma = np.array([[0.0, 0.0],[0.0, 0.0]], dtype=np.complex128)
#     dSigma_dgammac = np.array([[0.0, 0.0],[0.0, 0.0]], dtype=np.complex128)
#     dSigma_dphi = np.array([[2j*np.exp(1j*(2*phi+delta))*np.tanh(r), -1j*np.exp(1j*phi)/np.cosh(r)],
#                             [-1j*np.exp(1j*phi)/np.cosh(r), 0.0]])
#     dSigma_dr = np.array([[np.exp(1j*(2*phi+delta))/np.cosh(r)**2, np.exp(1j*phi)*np.tanh(r)/np.cosh(r)],
#                           [np.exp(1j*phi)*np.tanh(r)/np.cosh(r), -np.exp(-1j*delta)/np.cosh(r)**2]])
#     dSigma_ddelta = np.array([[1j*np.exp(1j*(2*phi+delta))*np.tanh(r), 0.0],
#                               [0.0, 1j*np.exp(-1j*delta)*np.tanh(r)]])
#     if r > 0.01:
#         dSigma_ddelta_over_r = dSigma_ddelta/r
#     else: # Taylor series for tanh(r)/r
#         dSigma_ddelta_over_r = np.array([[1j*np.exp(1j*(2*phi+delta))*(1 - r**2/3. + 2*r**4/15.), 0.0],
#                               [0.0, 1j*np.exp(-1j*delta)*(1 - r**2/3. + 2*r**4/15.)]])
#     dSigma_dz = np.exp(-1j*delta)*(dSigma_dr - 1j*dSigma_ddelta_over_r)
#     dSigma_dzc = np.exp(1j*delta)*(dSigma_dr + 1j*dSigma_ddelta_over_r)
#     dSigma = np.stack((dSigma_dgamma, dSigma_dgammac, dSigma_dphi, dSigma_dz, dSigma_dzc), axis=-1)

    return dC, dmu, dSigma


@jit(nopython=True)
def R_matrix(gamma, phi, theta1, varphi1, zeta, theta, varphi, Psi):
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        
        theta1(float): transmissivity angle of the beamsplitter1
        varphi1(float): reflection phase of the beamsplitter1
        
        zeta (complex np.array): squeezing parameter
        
        theta(float): transmissivity angle of the beamsplitter
        varphi(float): reflection phase of the beamsplitter
        
        old_state (np.array(complex)): State to be transformed

    Returns:
        R (complex array[D,D,D,D]): the matrix where R[:,:,0,0] is the transformed state 
    """
    C, mu ,Sigma = C_mu_Sigma(gamma, phi, theta1, varphi1, zeta, theta, varphi, dtype = np.complex128)

    cutoff = Psi.shape[0]
    dtype = Psi.dtype

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
            R[0,0,j,k] = np.sum(G_00pq2*Psi)

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


                    
    return R


@jit(nopython=True)
def G_matrix(gamma, phi, theta1, varphi1, zeta, theta, varphi, cutoff, dtype = np.complex128):
    """
    Directly constructs the transformation G matrix recursively and exactly.

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        
        theta1(float): transmissivity angle of the beamsplitter1
        varphi1(float): reflection phase of the beamsplitter1
        
        zeta (complex np.array): squeezing parameter
        
        theta(float): transmissivity angle of the beamsplitter
        varphi(float): reflection phase of the beamsplitter
        
        cutoff (int): Fock space cutoff dimension
        dtype (numpy dtype): dtype of the output

    Returns:
        (np.array(complex)): the transformation matrix G
    """
    
    C, mu ,Sigma = C_mu_Sigma(gamma, phi, theta1, varphi1, zeta, theta, varphi, dtype = dtype)

    sqrt = np.sqrt(np.arange(cutoff),dtype = dtype)
    G = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype = dtype)
    
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



@jit(nopython = True)
def grad_newstate(gamma, phi, theta1, varphi1, zeta, theta, varphi, Psi, G00, R):
    """
    Computes the gradient of the new state with respect to
    gamma, gamma*, phi, z, z* but not with respect to the old state

    Arguments:
        gamma (complex np.array): displacement parameter
        phi (float np.array): phase rotation parameter
        
        theta1(float): transmissivity angle of the beamsplitter1
        varphi1(float): reflection phase of the beamsplitter1
        
        zeta (complex np.array): squeezing parameter
        
        theta(float): transmissivity angle of the beamsplitter
        varphi(float): reflection phase of the beamsplitter
        
        Psi: (complex array): old state
        G00 (complex array[D]): G[0,0,:,:] of the G matrix 
        R (complex array[D,D]): complete R matrix R[:,:,:,:] (!not really complete....)

    Returns:
        (complex array[14, cutoff]): gradient of the new state with respect to
                                    gamma, phi, theta1, varphi1, zeta, theta, varphi
    """
    
    C, mu, Sigma = C_mu_Sigma(gamma, phi, theta1, varphi1, zeta, theta, varphi)
    dC, dmu, dSigma = dC_dmu_dSigma(gamma, phi, theta1, varphi1, zeta, theta, varphi)
    
    cutoff = len(Psi)
    dtype = Psi.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    
    dR = np.zeros((cutoff, cutoff, cutoff , cutoff, 12), dtype=dtype)
    dG_00pq = np.zeros((cutoff, cutoff, 12), dtype=dtype)
    
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
            dR[0,0,j,k] = np.sum(dG_00pq2*Psi)

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
