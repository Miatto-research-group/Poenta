import numpy as np
from numba import jit

@jit(nopython=True)
def C_(gamma, phi, z):
    """
    C constant (00 element of the Gaussian transformation matrix)

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
    """
    r = np.abs(z)
    delta = np.angle(z)
    
    return np.exp(-0.5*np.abs(gamma)**2 - 0.5*np.conj(gamma)**2*np.exp(1j*(2*phi+delta))*np.tanh(r))/np.sqrt(np.cosh(r))

@jit(nopython=True)
def dC_(gamma, phi, zeta):
    r = np.abs(zeta)
    delta = np.angle(zeta)
    
    C = C_(gamma, phi, zeta)
    dC_dgamma = (-0.5*np.conj(gamma))*C
    dC_dgammac = (-0.5*gamma - np.conj(gamma)*np.exp(1j*(2*phi+delta))*np.tanh(r))*C
    dC_dphi = (-1j*np.conj(gamma)**2*np.exp(1j*(2*phi+delta))*np.tanh(r))*C
    dC_dr = (- 0.5*np.conj(gamma)**2*np.exp(1j*(2*phi+delta))/np.cosh(r)**2)*C - 0.5*np.tanh(r)*C
    dC_ddelta = (- 0.5j*np.conj(gamma)**2*np.exp(1j*(2*phi+delta))*np.tanh(r))*C
    if r > 0.01:
        dC_ddelta_over_r = dC_ddelta/r
    else: # Taylor series for tanh(r)/r
        dC_ddelta_over_r = (- 0.5j*np.conj(gamma)**2*np.exp(1j*(2*phi+delta))*(1 - r**2/3. + 2*r**4/15.))*C
    dC_dz = np.exp(-1j*delta)*(dC_dr - 1j*dC_ddelta_over_r)
    dC_dzc = np.exp(1j*delta)*(dC_dr + 1j*dC_ddelta_over_r)
    
    return np.array([dC_dgamma, dC_dgammac, dC_dphi, dC_dz, dC_dzc])

@jit(nopython=True)
def mu_(gamma, phi, zeta):
    """
    Mu vector

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        zeta (complex): squeezing parameter
    """
    r = np.abs(zeta)
    delta = np.angle(zeta)
    return np.array([np.conj(gamma)*np.exp(1j*(2*phi+delta))*np.tanh(r) + gamma, -np.conj(gamma)*np.exp(1j*phi)/np.cosh(r)])

@jit(nopython=True)
def dmu_(gamma, phi, zeta):

    r = np.abs(zeta)
    delta = np.angle(zeta)
    dmu_dgamma = np.array([1.0, 0.0], dtype=np.complex128)
    dmu_dgammac = np.array([np.exp(1j*(2*phi+delta))*np.tanh(r), -np.exp(1j*phi)/np.cosh(r)])
    dmu_dphi = np.array([2j*np.conj(gamma)*np.exp(1j*(2*phi+delta))*np.tanh(r), -1j*np.exp(1j*phi)/np.cosh(r)])
    dmu_dr = np.array([np.conj(gamma)*np.exp(1j*(2*phi+delta))/np.cosh(r)**2, np.conj(gamma)*np.exp(1j*phi)*np.tanh(r)/np.cosh(r)])
    dmu_ddelta = np.array([1j*np.conj(gamma)*np.exp(1j*(2*phi+delta))*np.tanh(r), 0.0])
    if r > 0.01:
        dmu_ddelta_over_r = dmu_ddelta/r
    else: # Taylor series for tanh(r)/r
        dmu_ddelta_over_r = np.array([1j*np.conj(gamma)*np.exp(1j*(2*phi+delta))*(1 - r**2/3. + 2*r**4/15.), 0.0])
    dmu_dz = np.exp(-1j*delta)*(dmu_dr - 1j*dmu_ddelta_over_r)
    dmu_dzc = np.exp(1j*delta)*(dmu_dr + 1j*dmu_ddelta_over_r)
    
    return np.stack((dmu_dgamma, dmu_dgammac, dmu_dphi, dmu_dz, dmu_dzc), axis=-1)
    
@jit(nopython=True)
def Sigma_(gamma, phi, zeta):
    """
    Sigma matrix

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        zeta (complex): squeezing parameter
    """
    r = np.abs(zeta)
    delta = np.angle(zeta)
    
    return np.array([[np.exp(1j*(2*phi+delta))*np.tanh(r), -np.exp(1j*phi)/np.cosh(r)],
                     [-np.exp(1j*phi)/np.cosh(r), -np.exp(-1j*delta)*np.tanh(r)]])

@jit(nopython=True)
def dSigma_(gamma, phi, zeta):
    r = np.abs(zeta)
    delta = np.angle(zeta)
    
    dSigma_dgamma = np.array([[0.0, 0.0],[0.0, 0.0]], dtype=np.complex128)
    dSigma_dgammac = np.array([[0.0, 0.0],[0.0, 0.0]], dtype=np.complex128)
    dSigma_dphi = np.array([[2j*np.exp(1j*(2*phi+delta))*np.tanh(r), -1j*np.exp(1j*phi)/np.cosh(r)],
                            [-1j*np.exp(1j*phi)/np.cosh(r), 0.0]])
    dSigma_dr = np.array([[np.exp(1j*(2*phi+delta))/np.cosh(r)**2, np.exp(1j*phi)*np.tanh(r)/np.cosh(r)],
                          [np.exp(1j*phi)*np.tanh(r)/np.cosh(r), -np.exp(-1j*delta)/np.cosh(r)**2]])
    dSigma_ddelta = np.array([[1j*np.exp(1j*(2*phi+delta))*np.tanh(r), 0.0],
                              [0.0, 1j*np.exp(-1j*delta)*np.tanh(r)]])
    if r > 0.01:
        dSigma_ddelta_over_r = dSigma_ddelta/r
    else: # Taylor series for tanh(r)/r
        dSigma_ddelta_over_r = np.array([[1j*np.exp(1j*(2*phi+delta))*(1 - r**2/3. + 2*r**4/15.), 0.0],
                              [0.0, 1j*np.exp(-1j*delta)*(1 - r**2/3. + 2*r**4/15.)]])
    dSigma_dz = np.exp(-1j*delta)*(dSigma_dr - 1j*dSigma_ddelta_over_r)
    dSigma_dzc = np.exp(1j*delta)*(dSigma_dr + 1j*dSigma_ddelta_over_r)
    
    return np.stack((dSigma_dgamma, dSigma_dgammac, dSigma_dphi, dSigma_dz, dSigma_dzc), axis=-1)

@jit(nopython = True)
def new_state(gamma, phi, z, old_state):
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (np.array(complex)): State to be transformed

    Returns:
        (np.array(complex)): the transformed state
    """
    C = C_(gamma, phi, z)
    mu = mu_(gamma, phi, z)
    Sigma = Sigma_(gamma, phi, z)
    
    cutoff = old_state.shape[0]
    dtype = old_state.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    
    R = np.zeros((cutoff, cutoff), dtype=dtype)
    G = np.zeros(cutoff, dtype=dtype)
    
    # first row of Transformation matrix
    G[0] = C
    for n in range(1, cutoff):
        G[n] = mu[1]/sqrt[n]*G[n-1] - Sigma[1,1]*sqrt[n-1]/sqrt[n]*G[n-2]
    
    # first row of R matrix
    for n in range(cutoff):
        R[0, n] = np.dot(G[:cutoff - n], old_state)
        old_state = old_state[1:]*sqrt[1:cutoff-n]

    
    # rest of R matrix
    for m in range(1, cutoff):
        for n in range(cutoff-m): 
            R[m, n] = mu[0]/sqrt[m]*R[m-1, n] - Sigma[0,0]*sqrt[m-1]/sqrt[m]*R[m-2, n] - Sigma[0,1]/sqrt[m]*R[m-1, n+1]
            
    return R[:, 0]


@jit(nopython = True)
def approx_new_state(gamma, phi, z, old_state, order = None):
    """
    Constructs the transformed state recursively and exactly
    up to the Nth Fock amplitude, indicated by the keyword argument `order`

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (np.array(complex)): State to be transformed
        order (int): Fock space dimensionality of the exact approximation

    Returns:
        (np.array(complex)): the new state which is exact up to dimension `order`

    """
    C = C_(gamma, phi, z)
    mu = mu_(gamma, phi, z)
    Sigma = Sigma_(gamma, phi, z)
    
    cutoff = old_state.shape[0]
    dtype = old_state.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    if order is None:
        order = cutoff
    
    R = np.zeros((cutoff, cutoff), dtype=dtype)
    G = np.zeros(cutoff, dtype=dtype)
    
    # first row of Transformation matrix
    G[0] = C
    for n in range(1, cutoff):
        G[n] = mu[1]/sqrt[n]*G[n-1] - Sigma[1,1]*sqrt[n-1]/sqrt[n]*G[n-2]
    
    # first row of R matrix
    for n in range(order):
        R[0, n] = np.dot(G[:cutoff - n], old_state)
        old_state = old_state[1:]*sqrt[1:cutoff-n]

    # rest of R matrix
    for m in range(1, cutoff):
        for n in range(max(1, order-m)): 
            R[m, n] = mu[0]/sqrt[m]*R[m-1, n] - Sigma[0,0]*sqrt[m-1]/sqrt[m]*R[m-2, n] - Sigma[0,1]/sqrt[m]*R[m-1, n+1]
            
    return R[:, 0]

@jit(nopython = True)
def G_matrix(gamma, phi, zeta, cutoff, dtype=np.complex128):
    """
    Constructs the Gaussian transformation recursively

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        zeta (complex): squeezing parameter
        cutoff (int): Fock space cutoff dimension
        dtype (numpy dtype): dtype of the output

    Returns:
        G (np.array(complex)): the single-mode Gaussian transformation matrix
    """
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    G = np.zeros((cutoff, cutoff), dtype=dtype)
    
    C = C_(gamma, phi, zeta)
    mu = mu_(gamma, phi, zeta)
    Sigma = Sigma_(gamma, phi, zeta)
    
    # First column 
    G[0,0] = C
    for m in range(cutoff-1):
        G[m+1,0] = mu[0]/sqrt[m+1]*G[m,0] - Sigma[0,0]*sqrt[m]/sqrt[m+1]*G[m-1,0]
    
    # All rows
    for m in range(cutoff):
        for n in range(cutoff-1): 
            G[m, n+1] = mu[1]/sqrt[n+1]*G[m, n] - Sigma[1,0]*sqrt[m]/sqrt[n+1]*G[m-1, n] - Sigma[1,1]*sqrt[n]/sqrt[n+1]*G[m, n-1]
            
    return G

@jit(nopython = True)
def RG(C, mu, Sigma, old_state):
    """
    Constructs the R and G matrices (here G matrices is not the same as the one above!!)
    
    Arguments:
        C (complex): C constant
        mu (np.array(complex)): mu vector
        Sigma (np.array(complex)): Sigma matrix
        old_state (np.array(complex)): input state

    Returns:
        R, G (np.array(complex), np.array(complex)): The R and G matrices

    """
    
    cutoff = old_state.shape[0]
    dtype = old_state.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    
    R = np.zeros((cutoff, cutoff), dtype=dtype)
    G = np.zeros(cutoff, dtype=dtype)
    
    # first row of Transformation matrix
    G[0] = C
    for n in range(1, cutoff):
        G[n] = mu[1]/sqrt[n]*G[n-1] - Sigma[1,1]*sqrt[n-1]/sqrt[n]*G[n-2]
    
    # first row of R matrix
    for n in range(cutoff):
        R[0, n] = np.dot(G[:cutoff - n], old_state)
        old_state = old_state[1:]*sqrt[1:cutoff-n]

    
    # rest of Auxiliary matrix
    for m in range(1, cutoff):
        for n in range(cutoff-m): 
            R[m, n] = mu[0]/sqrt[m]*R[m-1, n] - Sigma[0,0]*sqrt[m-1]/sqrt[m]*R[m-2, n] - Sigma[0,1]/sqrt[m]*R[m-1, n+1]
            
    return R, G