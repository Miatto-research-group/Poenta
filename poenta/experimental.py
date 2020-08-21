#     Copyright (C) 2020 Miatto research group.

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numpy import expand_dims as ed
from numba import njit
from .jitted import C_mu_Sigma, dC_dmu_dSigma, convert_scalar, C_mu_Sigma2, dC_dmu_dSigma2,


@njit()
def R_matrix(gamma, phi, z, old_state):
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (complex array[batch, D]): State to be transformed

    Returns:
        R (complex array[batch,D,D]): the matrix whose 1st column is the transformed state
    """
    batch, cutoff = old_state.shape
    dtype = old_state.dtype

    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)

    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    R = np.zeros((batch, cutoff, cutoff), dtype=dtype)
    G0 = np.zeros(cutoff, dtype=dtype)

    # first row of Transformation matrix
    G0[0] = C
    for n in range(1, cutoff):
        G0[n] = mu[1] / sqrt[n] * G0[n - 1] - Sigma[1, 1] * sqrt[n - 1] / sqrt[n] * G0[n - 2]

    # first row of R matrix
    for n in range(cutoff):
        R[:, 0, n] = np.sum(old_state * G0[: cutoff - n], axis=-1)
        old_state = old_state[:, 1:] * sqrt[1 : cutoff - n]

    # second row of R matrix
    R[:, 1, :-1] = mu[0] * R[:, 0, :-1] - Sigma[0, 1] * R[:, 0, 1:]
        
    # rest of R matrix
    for m in range(2, cutoff):
        R[:, m, :-m] = (
            mu[0] * R[:, m - 1, :-m]
            - Sigma[0, 0] * sqrt[m - 1] * R[:, m - 2, :-m]
            - Sigma[0, 1] * R[:, m - 1, 1 : -m + 1]
        ) / sqrt[m]

    return R
    
@njit()
def R_matrix2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, old_state):
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma1 (complex): displacement parameter1
        gamma2 (complex): displacement parameter2
        phi1 (float): phase rotation parameter1
        phi2 (float): phase rotation parameter2
        
        theta1(float): transmissivity angle of the beamsplitter1
        varphi1(float): reflection phase of the beamsplitter1
        
        zeta1 (complex): squeezing parameter1
        zeta2 (complex): squeezing parameter2
        
        theta(float): transmissivity angle of the beamsplitter
        varphi(float): reflection phase of the beamsplitter
        
        Psi(np.array(complex)): State to be transformed

    Returns:
        R (complex array[batch,D,D,D,D]): the matrix where R[batch,:,:,0,0] is the transformed state for each batch
    """
    
    batch, cutoff = old_state.shape[0], old_state.shape[1]
    dtype = old_state.dtype
    
    gamma1 = convert_scalar(gamma1)
    gamma2 = convert_scalar(gamma2)
    phi1 = convert_scalar(phi1)
    phi2 = convert_scalar(phi2)
    theta1 = convert_scalar(theta1)
    varphi1 = convert_scalar(varphi1)
    zeta1 = convert_scalar(zeta1)
    zeta2 = convert_scalar(zeta2)
    theta = convert_scalar(theta)
    varphi = convert_scalar(varphi)
    
    C, mu ,Sigma = C_mu_Sigma2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi)

    sqrt = np.sqrt(np.arange(cutoff, dtype = dtype))
    sqrtT = sqrt.reshape(-1, 1)

    R = np.zeros((batch, cutoff, cutoff, cutoff+1, cutoff+1), dtype = dtype)
    G_00pq = np.zeros((cutoff, cutoff), dtype = dtype)
    
    
    #G_mn00
    G_00pq[0,0] = C
    for q in range(1, cutoff):
        G_00pq[0,q] = (mu[3]*G_00pq[0,q-1] - Sigma[3,3]*sqrt[q-1]*G_00pq[0,q-2])/sqrt[q]


    for p in range(1,cutoff):
        for q in range(0,cutoff):
            G_00pq[p,q] = (mu[2]*G_00pq[p-1,q] - Sigma[2,2]*sqrt[p-1]*G_00pq[p-2,q] - Sigma[2,3]*sqrt[q]*G_00pq[p-1,q-1])/sqrt[p]
                    
    # R_00^jk = a_dagger^j \G_00pq> b^k  * |old_state>
    G_00pq2 = G_00pq
    for j in range(cutoff):
        G_00pq3 = G_00pq2
        for k in range(cutoff):
            R[:,0,0,j,k] = np.sum(G_00pq3*Psi[j:,k:])
            G_00pq3 = G_00pq3[:,:-1]*sqrt[k+1:]
        G_00pq2 = sqrtT[j+1:]*G_00pq2[:-1,:]

    #R_0n^jk
    for n in range(1,cutoff):
        for k in range(0,cutoff):
            for j in range(0,cutoff):
                R[:,0,n,j,k] = mu[1]/sqrt[n]*R[0,n-1,j,k] - Sigma[1,1]/sqrt[n]*sqrt[n-1]*R[0,n-2,j,k] - Sigma[1,2]/sqrt[n]*R[0,n-1,j+1,k] - Sigma[1,3]/sqrt[n]*R[0,n-1,j,k+1]

    #R_mn^jk
    for m in range(1,cutoff):
        for n in range(0,cutoff):
            for j in range(0,cutoff-m):
                for k in range(0,cutoff-m-j):
                    R[:,m,n,j,k] = mu[0]/sqrt[m]*R[m-1,n,j,k] - Sigma[0,0]/sqrt[m]*sqrt[m-1]*R[m-2,n,j,k] - Sigma[0,1]*sqrt[n]/sqrt[m]*R[m-1,n-1,j,k] - Sigma[0,2]/sqrt[m]*R[m-1,n,j+1,k] - Sigma[0,3]/sqrt[m]*R[m-1,n,j,k+1]
          
    return R


@njit()
def dPsi(gamma: np.complex, phi: np.float, z: np.complex, state_in: np.array, G0: np.array, R: np.array) -> list:

    batch, cutoff = state_in.shape
    dtype = state_in.dtype

    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)

    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)
    dC, dmu, dSigma = dC_dmu_dSigma(gamma, phi, z)

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    dR = np.zeros((batch, cutoff, cutoff, 5), dtype=dtype)
    dG0 = np.zeros((cutoff, 5), dtype=dtype)

    # first row of Transformation matrix
    dG0[0] = dC
    for n in range(1, cutoff):
        dG0[n] = (
            dmu[1] / sqrt[n] * G0[n - 1]
            - dSigma[1, 1] * sqrt[n - 1] / sqrt[n] * G0[n - 2]
            + mu[1] / sqrt[n] * dG0[n - 1]
            - Sigma[1, 1] * sqrt[n - 1] / sqrt[n] * dG0[n - 2]
        )

    # first row of dR matrix
    for n in range(cutoff):
        dR[:, 0, n] = np.dot(state_in, dG0[: cutoff - n])
        state_in = state_in[:, 1:] * sqrt[1 : cutoff - n]

    # second row of dR matrix
    dR[:, 1, :-1] = (
        ed(R[:, 0, :-1], 2) * ed(ed(dmu[0], 0), 1)
        - ed(R[:, 0, 1:], 2) * ed(ed(dSigma[0, 1], 0), 0)
        + mu[0] * dR[:, 0, :-1]
        - Sigma[0, 1] * dR[:, 0, 1:]
    )
    
    # rest of dR matrix
    for m in range(2, cutoff):
        dR[:, m, :-m] = (
            ed(R[:, m - 1, :-m], 2) * ed(ed(dmu[0], 0), 1)
            - sqrt[m - 1] * ed(R[:, m - 2, :-m], 2) * ed(ed(dSigma[0, 0], 0), 0)
            - ed(R[:, m - 1, 1 : -m + 1], 2) * ed(ed(dSigma[0, 1], 0), 0)
            + mu[0] * dR[:, m - 1, :-m]
            - Sigma[0, 0] * sqrt[m - 1] * dR[:, m - 2, :-m]
            - Sigma[0, 1] * dR[:, m - 1, 1 : -m + 1]
        ) / sqrt[m]

    return list(np.transpose(dR[:, :, 0], (2, 0, 1)))

@njit()
def dPsi2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, state_in, G00, R):
    """
    Computes the gradient of the new state with respect to
    gamma, gamma*, phi, z, z* but not with respect to the old state

    Arguments:
        gamma1 (complex): displacement parameter1
        gamma2 (complex): displacement parameter2
        phi1 (float): phase rotation parameter1
        phi2 (float): phase rotation parameter2
        
        theta1(float): transmissivity angle of the beamsplitter1
        varphi1(float): reflection phase of the beamsplitter1
        
        zeta1 (complex): squeezing parameter1
        zeta2 (complex): squeezing parameter2
        
        theta(float): transmissivity angle of the beamsplitter
        varphi(float): reflection phase of the beamsplitter
        
        Psi: (complex array): old state
        G00 (complex array[D]): G[0,0,:,:] of the G matrix
        R (complex array[D,D]): complete R matrix R[:,:,:,:] (!not really complete....)

    Returns:
        (complex array[cutoff, cutoff, 14]): gradient of the new state with respect to
                                    gamma1, gamma1*, gamma2, gamma2*, phi1, phi2, theta1, varphi1, zeta1, zeta1*, zeta2, zeta2*, theta, varphi
    """
    batch, cutoff = state_in.shape
    dtype = state_in.dtype
    
    C, mu, Sigma = C_mu_Sigma2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi)
    dC, dmu, dSigma = dC_dmu_dSigma2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi)
    
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    sqrtT = sqrt.reshape(-1, 1)
    
    dR = np.zeros((cutoff, cutoff, cutoff+1 , cutoff+1, 14),dtype = dtype)
    dG00 = np.zeros((cutoff, cutoff, 14),dtype = dtype)
    
    dG00[0,0] = dC
    for q in range(1, cutoff):
        dG00[0,q] = (dmu[3]*G00[0,q-1]+mu[3]*dG00[0,q-1] - dSigma[3,3]*sqrt[q-1]*G00[0,q-2]- Sigma[3,3]*sqrt[q-1]*dG00[0,q-2])/sqrt[q]


    for p in range(1,cutoff):
        for q in range(0,cutoff):
            dG00[p,q] = (dmu[2]*G00[p-1,q]+ mu[2]*dG00[p-1,q] - dSigma[2,2]*sqrt[p-1]*G00[p-2,q]- Sigma[2,2]*sqrt[p-1]*dG00[p-2,q] - dSigma[2,3]*sqrt[q]*G00[p-1,q-1]- Sigma[2,3]*sqrt[q]*dG00[p-1,q-1])/sqrt[p]
                    
    dG002 = dG00
    for j in range(cutoff):
        dG003 = dG002
        for k in range(cutoff):
            dR[0,0,j,k] = np.sum(dG003*Psi[j:,k:])
            dG003 = dG003[:,:-1]*sqrt[k+1:]
        dG002 = sqrtT[j+1:]*dG002[:-1,:]
            

    for n in range(1,cutoff):
        for k in range(0,cutoff):
            for j in range(0,cutoff):
                dR[0,n,j,k] = (dmu[1]*R[0,n-1,j,k] + mu[1]*dR[0,n-1,j,k] - dSigma[1,1]*sqrt[n-1]*R[0,n-2,j,k] - Sigma[1,1]*sqrt[n-1]*dR[0,n-2,j,k] - dSigma[1,2]*R[0,n-1,j+1,k] - Sigma[1,2]*dR[0,n-1,j+1,k] - dSigma[1,3]*R[0,n-1,j,k+1] - Sigma[1,3]*dR[0,n-1,j,k+1])/sqrt[n]


    for m in range(1,cutoff):
        for n in range(0,cutoff):
            for j in range(0,cutoff-m):
                for k in range(0,cutoff-m-j):
                    dR[m,n,j,k] = (dmu[0]*R[m-1,n,j,k] + mu[0]*dR[m-1,n,j,k] - dSigma[0,0]*sqrt[m-1]*R[m-2,n,j,k] - Sigma[0,0]*sqrt[m-1]*dR[m-2,n,j,k] - dSigma[0,1]*sqrt[n]*R[m-1,n-1,j,k] - Sigma[0,1]*sqrt[n]*dR[m-1,n-1,j,k] - dSigma[0,2]*R[m-1,n,j+1,k] - Sigma[0,2]*dR[m-1,n,j+1,k] - dSigma[0,3]*R[m-1,n,j,k+1] - Sigma[0,3]*dR[m-1,n,j,k+1])/sqrt[m]
           
    return np.transpose(dR[:,:,0,0])
