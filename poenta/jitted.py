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
from numba import njit
import numba as nb


@nb.generated_jit
def convert_scalar(arr):
    """ helper function that turns 0d-arrays into scalars """
    if isinstance(arr, nb.types.Array) and arr.ndim == 0:
        return lambda arr: arr[()]
    else:
        return lambda arr: arr


@njit  # (nb.types.Tuple((nb.complex128, nb.complex128[:], nb.complex128[:,:]))(nb.complex128, nb.float64, nb.complex128))
def C_mu_Sigma(gamma: np.complex, phi: np.float, z: np.complex) -> tuple:
    """
    Utility function to construct:
    1. C constant
    2. Mu vector
    3. Sigma matrix
    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        dtype (numpy type): unused for now
    Returns:
        C (complex), mu (complex array[2]), Sigma (complex array[2,2])
    """
    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)
    r = np.abs(z)
    delta = np.angle(z)
    exp2phidelta = np.exp(1j * (2 * phi + delta))
    eiphi = np.exp(1j * phi)
    tanhr = np.tanh(r)
    coshr = np.cosh(r)
    cgamma = np.conj(gamma)

    C = np.exp(-0.5 * np.abs(gamma) ** 2 - 0.5 * cgamma ** 2 * exp2phidelta * tanhr) / np.sqrt(coshr)
    mu = np.array([cgamma * exp2phidelta * tanhr + gamma, -cgamma * eiphi / coshr,])
    Sigma = np.array([[exp2phidelta * tanhr, -eiphi / coshr], [-eiphi / coshr, -np.exp(-1j * delta) * tanhr],])

    return C, mu, Sigma
    
@njit
def C_mu_Sigma2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi):
    """
    Utility function to construct:
    1. C constant
    2. Mu vector
    3. Sigma matrix

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

    Returns:
        C (complex), mu (complex array[4]), Sigma (complex array[4,4])
    """
   
    
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

    gamma1c = np.conj(gamma1)
    gamma2c = np.conj(gamma2)
    r1 = np.abs(zeta1)
    r2 = np.abs(zeta2)
    delta1 = np.angle(zeta1)
    delta2 = np.angle(zeta2)
    
    e_iphi1 = np.exp(1j*phi1)
    e_iphi2 = np.exp(1j*phi2)
    e_iphi12 = np.exp(1j*(phi1+phi2))
    e_ivarphi1 = np.exp(1j*varphi1)
    e_ivarphi = np.exp(1j*varphi)
    e_idelta1 = np.exp(1j*delta1)
    e_idelta2 = np.exp(1j*delta2)
    
    cos_theta1 = np.cos(theta1)
    sin_theta1 = np.sin(theta1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    tanh_r1 = np.tanh(r1)
    tanh_r2 = np.tanh(r2)
    cosh_r1 = np.cosh(r1)
    cosh_r2 = np.cosh(r2)
    
    T1 = np.exp(1j*delta1)*np.tanh(r1)
    T2 = np.exp(1j*delta2)*np.tanh(r2)
    T1minus = np.exp(-1j*delta1)*np.tanh(r1)
    T2minus = np.exp(-1j*delta2)*np.tanh(r2)

    W = np.array([[e_iphi1*cos_theta1 , -e_iphi1/e_ivarphi1*sin_theta1],[e_iphi2*e_ivarphi1*sin_theta1, e_iphi2*cos_theta1]] ,dtype = np.complex128)
    V = np.array([[cos_theta , -1/e_ivarphi*sin_theta],[e_ivarphi*sin_theta, cos_theta]],dtype = np.complex128)

    WdiagWT = np.array([[e_idelta1*e_iphi1**2*cos_theta1**2*tanh_r1 + e_idelta2*e_iphi1**2/e_ivarphi1**2*sin_theta1**2*tanh_r2, e_idelta1*e_iphi12*e_ivarphi1*cos_theta1*sin_theta1*tanh_r1 - e_idelta2*e_iphi12/e_ivarphi1*cos_theta1*sin_theta1*tanh_r2],[ e_idelta1*e_iphi12*e_ivarphi1*cos_theta1*sin_theta1*tanh_r1 - e_idelta2*e_iphi12/e_ivarphi1*cos_theta1*sin_theta1*tanh_r2, e_idelta1*e_iphi2**2*e_ivarphi1**2*sin_theta1**2*tanh_r1+e_idelta2*e_iphi2**2*cos_theta1**2*tanh_r2]],dtype = np.complex128)
    
    WdiagsV = np.array([[ e_iphi1*cos_theta1*cos_theta/cosh_r1 - e_iphi1*e_ivarphi/e_ivarphi1*sin_theta*sin_theta1/cosh_r2 ,  -e_iphi1/e_ivarphi*cos_theta1*sin_theta/cosh_r1 - e_iphi1/e_ivarphi1*cos_theta*sin_theta1/cosh_r2],[e_iphi2*e_ivarphi*cos_theta1*sin_theta/cosh_r2 + e_iphi2*e_ivarphi1*cos_theta*sin_theta1/cosh_r1     ,  e_iphi2*cos_theta1*cos_theta/cosh_r2 - e_iphi2*e_ivarphi1/e_ivarphi*sin_theta*sin_theta1/cosh_r1   ]],dtype = np.complex128)
    
    VTdiagminusV = np.array([[1/e_idelta1*cos_theta**2*tanh_r1 + 1/e_idelta2*e_ivarphi**2*sin_theta**2*tanh_r2, -1/e_idelta1/e_ivarphi*cos_theta*sin_theta*tanh_r1 + 1/e_idelta2*e_ivarphi*sin_theta*cos_theta*tanh_r2],[ -1/e_idelta1/e_ivarphi*cos_theta*sin_theta*tanh_r1 + 1/e_idelta2*e_ivarphi*sin_theta*cos_theta*tanh_r2, 1/e_idelta1/e_ivarphi**2*sin_theta**2*tanh_r1 + 1/e_idelta2*cos_theta**2*tanh_r2]],dtype = np.complex128)
 
    Cpart2 = (gamma1c*WdiagWT[0,0] + gamma2c * WdiagWT[1,0]) * gamma1c + (gamma1c*WdiagWT[0,1] + gamma2c * WdiagWT[1,1]) * gamma2c
    C = np.exp(-0.5*(np.abs(gamma1)**2+np.abs(gamma2)**2 +Cpart2))/ np.sqrt(cosh_r1*cosh_r2)
    
    mu = np.zeros(4, dtype = np.complex128)
    
    mu[0] = gamma1c* WdiagWT[0,0] + gamma2c*WdiagWT[1,0] + gamma1
    mu[1] = gamma1c* WdiagWT[0,1] + gamma2c*WdiagWT[1,1] + gamma2
    mu[2] = -(gamma1c* WdiagsV[0,0] + gamma2c*WdiagsV[1,0])
    mu[3] = -(gamma1c* WdiagsV[0,1] + gamma2c*WdiagsV[1,1])

    W1 = WdiagWT
    W2 = -WdiagsV
    W3 = np.transpose(W2)
    W4 = -VTdiagminusV
    Sigma = np.concatenate((np.concatenate( (W1,W2) ,axis=1),np.concatenate((W3, W4),axis=1)))
    
    return C, mu, Sigma


@njit
def dC_dmu_dSigma(gamma: np.complex, phi: np.float, z: np.complex) -> tuple:
    """
    Utility function to construct the gradient of:
    1. C constant
    2. Mu vector
    3. Sigma matrix
    with respect to gamma, gamma*, phi, z and z*
    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
    Returns:
        dC (complex array[5]), dmu (complex array[2,5]), dSigma (complex array[2,2,5])
    """
    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)
    r = np.abs(z)
    delta = np.angle(z)
    exp2phidelta = np.exp(1j * (2 * phi + delta))
    eidelta = np.exp(1j * delta)
    eideltac = np.exp(-1j * delta)
    eiphi = np.exp(1j * phi)
    tanhr = np.tanh(r)
    coshr = np.cosh(r)
    cgamma = np.conj(gamma)

    # dC
    dC_dgamma = (-0.5 * cgamma) * C
    dC_dgammac = (-0.5 * gamma - cgamma * exp2phidelta * tanhr) * C
    dC_dphi = (-1j * cgamma ** 2 * exp2phidelta * tanhr) * C
    dC_dr = (-0.5 * cgamma ** 2 * exp2phidelta / coshr ** 2) * C - 0.5 * tanhr * C
    dC_ddelta = (-0.5j * cgamma ** 2 * exp2phidelta * tanhr) * C
    if r > 0.01:
        dC_ddelta_over_r = dC_ddelta / r
    else:  # Taylor series for tanh(r)/r
        dC_ddelta_over_r = (-0.5j * cgamma ** 2 * exp2phidelta * (1 - r ** 2 / 3.0 + 2 * r ** 4 / 15.0)) * C
    dC_dz = eideltac * (dC_dr - 1j * dC_ddelta_over_r) / 2
    dC_dzc = eidelta * (dC_dr + 1j * dC_ddelta_over_r) / 2
    dC = np.array([dC_dgamma, dC_dgammac, dC_dphi, dC_dz, dC_dzc])

    # dmu
    dmu_dgamma = np.array([1.0, 0.0], dtype=np.complex128)
    dmu_dgammac = np.array([exp2phidelta * tanhr, -eiphi / coshr])
    dmu_dphi = np.array([2j * cgamma * exp2phidelta * tanhr, -1j * eiphi / coshr])
    dmu_dr = np.array([cgamma * exp2phidelta / coshr ** 2, cgamma * eiphi * tanhr / coshr,])
    dmu_ddelta = np.array([1j * cgamma * exp2phidelta * tanhr, 0.0])
    if r > 0.01:
        dmu_ddelta_over_r = dmu_ddelta / r
    else:  # Taylor series for tanh(r)/r
        dmu_ddelta_over_r = np.array([1j * cgamma * exp2phidelta * (1 - r ** 2 / 3.0 + 2 * r ** 4 / 15.0), 0.0])
    dmu_dz = eideltac * (dmu_dr - 1j * dmu_ddelta_over_r) / 2
    dmu_dzc = eidelta * (dmu_dr + 1j * dmu_ddelta_over_r) / 2
    dmu = np.stack((dmu_dgamma, dmu_dgammac, dmu_dphi, dmu_dz, dmu_dzc), axis=-1)

    # dSigma
    dSigma_dgamma = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    dSigma_dgammac = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    dSigma_dphi = np.array([[2j * exp2phidelta * tanhr, -1j * eiphi / coshr], [-1j * eiphi / coshr, 0.0],])
    dSigma_dr = np.array(
        [[exp2phidelta / coshr ** 2, eiphi * tanhr / coshr], [eiphi * tanhr / coshr, -eideltac / coshr ** 2],]
    )
    dSigma_ddelta = np.array([[1j * exp2phidelta * tanhr, 0.0], [0.0, 1j * eideltac * tanhr]])
    if r > 0.01:
        dSigma_ddelta_over_r = dSigma_ddelta / r
    else:  # Taylor series for tanh(r)/r
        dSigma_ddelta_over_r = np.array(
            [
                [1j * exp2phidelta * (1 - r ** 2 / 3.0 + 2 * r ** 4 / 15.0), 0.0],
                [0.0, 1j * eideltac * (1 - r ** 2 / 3.0 + 2 * r ** 4 / 15.0)],
            ]
        )
    dSigma_dz = eideltac * (dSigma_dr - 1j * dSigma_ddelta_over_r) / 2
    dSigma_dzc = eidelta * (dSigma_dr + 1j * dSigma_ddelta_over_r) / 2
    dSigma = np.stack((dSigma_dgamma, dSigma_dgammac, dSigma_dphi, dSigma_dz, dSigma_dzc), axis=-1)

    return dC, dmu, dSigma

@njit
def dC_dmu_dSigma2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi):
    """
    Utility function to construct the gradient of:
    1. C constant
    2. Mu vector
    3. Sigma matrix
    with respect to gamma1, gamma1*, gamma2, gamma2*, phi1, phi2, theta1, varphi1, zeta1, zeta1*, zeta2, zeta2*, theta, varphi

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

    Returns:
       dC (complex array[14]), dmu (complex array[4,14]), dSigma (complex array[4,4,14])
    """

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

    C, mu, Sigma = C_mu_Sigma2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi)

    gamma1c = np.conj(gamma1)
    gamma2c = np.conj(gamma2)
    r1 = np.abs(zeta1)
    r2 = np.abs(zeta2)
    delta1 = np.angle(zeta1)
    delta2 = np.angle(zeta2)

    e_iphi1 = np.exp(1j*phi1)
    e_iphi2 = np.exp(1j*phi2)
    e_iphi12 = np.exp(1j*(phi1+phi2))
    e_ivarphi1 = np.exp(1j*varphi1)
    e_ivarphi = np.exp(1j*varphi)
    e_idelta1 = np.exp(1j*delta1)
    e_idelta2 = np.exp(1j*delta2)

    cos_theta1 = np.cos(theta1)
    sin_theta1 = np.sin(theta1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    tanh_r1 = np.tanh(r1)
    tanh_r2 = np.tanh(r2)
    cosh_r1 = np.cosh(r1)
    cosh_r2 = np.cosh(r2)

    T1 = np.exp(1j*delta1)*np.tanh(r1)
    T2 = np.exp(1j*delta2)*np.tanh(r2)
    T1minus = np.exp(-1j*delta1)*np.tanh(r1)
    T2minus = np.exp(-1j*delta2)*np.tanh(r2)


    # dC
    dC_dgamma1 = -C*gamma1c/2
    dC_dgamma1c = -C/2*(gamma1+2*gamma1c*e_iphi1**2*(cos_theta1**2*T1+1/e_ivarphi1**2*sin_theta1**2*T2) + gamma2c*e_iphi12*np.sin(2*theta1)*(e_ivarphi1*T1-1/e_ivarphi1*T2))
    dC_dgamma2 = -C*gamma2c/2
    dC_dgamma2c = -C/2*(gamma2+2*gamma2c*e_iphi2**2*(cos_theta1**2*T2+e_ivarphi1**2*sin_theta1**2*T1) + gamma1c*e_iphi12*np.sin(2*theta1)*(e_ivarphi1*T1-1/e_ivarphi1*T2))
    dC_dphi1 = -C/2*(2j*gamma1c**2*e_iphi1**2*(cos_theta1**2*T1+1/e_ivarphi1**2*sin_theta1**2*T2)+1j*gamma1c*gamma2c*e_iphi12*np.sin(2*theta1)*(e_ivarphi1*T1-1/e_ivarphi1*T2))
    dC_dphi2 = -C/2*(2j*gamma2c**2*e_iphi2**2*(cos_theta1**2*T2+e_ivarphi1**2*sin_theta1**2*T2)+1j*gamma1c*gamma2c*e_iphi12*np.sin(2*theta1)*(e_ivarphi1*T1-1/e_ivarphi1*T2))
    dC_dtheta1 = -C/2*(T1-1/e_ivarphi1**2*T2)*(-gamma1c**2*e_iphi1**2*np.sin(2*theta1)+2*gamma1c*gamma2c*e_iphi12*e_ivarphi1*np.cos(2*theta1)+gamma2c**2*e_iphi2**2*np.sin(2*theta1)*e_ivarphi1**2)
    dC_dvarphi1 = -1j*C*sin_theta1*(-gamma1c**2*e_iphi1**2/e_ivarphi1**2*sin_theta1*T2 + gamma1c*gamma2c*e_iphi12*cos_theta1*(e_ivarphi1*T1+1/e_ivarphi1*T2)+gamma2c**2*e_iphi2**2*e_ivarphi1**2*sin_theta1*T1 )

    dC_dr1 = -1/2*tanh_r1*C - C/2*e_idelta1/cosh_r1**2*(gamma1c*e_iphi1*cos_theta1 + gamma2c*e_iphi2*e_ivarphi1*sin_theta1)**2
    dC_ddelta1 = - 1j*C/2*e_idelta1*tanh_r1*(gamma1c*e_iphi1*cos_theta1 + gamma2c*e_iphi2*e_ivarphi1*sin_theta1)**2
    if r1 > 0.01:
        dC_ddelta1_over_r = dC_ddelta1/r1
    else: # Taylor series for tanh(r)/r
        dC_ddelta1_over_r = (1 - r1**2/3. + 2*r1**4/15.)*(-1/2*C - 1j*C/2*e_idelta1*(gamma1c*e_iphi1*cos_theta1+gamma2c*e_iphi2*e_ivarphi1*sin_theta1)**2)
    dC_dzeta1 = np.exp(-1j*delta1)*(dC_dr1 - 1j*dC_ddelta1_over_r)/2
    dC_dzeta1c = e_idelta1*(dC_dr1 + 1j*dC_ddelta1_over_r)/2

    dC_dr2 = -1/2*tanh_r2*C - C/2*np.exp(1j*delta2)/cosh_r2**2/e_ivarphi1**2*(-gamma1c*e_iphi1*sin_theta1+gamma2c*e_iphi2*e_ivarphi1*cos_theta1)**2
    dC_ddelta2 = - 1j*C/2*np.exp(1j*delta2)*tanh_r2/e_ivarphi1**2*(-gamma1c*e_iphi1*sin_theta1+gamma2c*e_iphi2*e_ivarphi1*cos_theta1)**2
    if r2 > 0.01:
        dC_ddelta2_over_r = dC_ddelta2/r2
    else: # Taylor series for tanh(r)/r
        dC_ddelta2_over_r = (1 - r2**2/3. + 2*r2**4/15.)*(-1/2*C - 1j*C/2*np.exp(1j*delta2)/e_ivarphi1**2*(-gamma1c*e_iphi1*sin_theta1+gamma2c*e_iphi2*e_ivarphi1*cos_theta1)**2)
    dC_dzeta2 = np.exp(-1j*delta2)*(dC_dr2 - 1j*dC_ddelta2_over_r)/2
    dC_dzeta2c = np.exp(1j*delta2)*(dC_dr2 + 1j*dC_ddelta2_over_r)/2

    dC_dtheta = 0
    dC_dvarphi = 0

    dC = np.array([dC_dgamma1, dC_dgamma1c, dC_dgamma2, dC_dgamma2c, dC_dphi1, dC_dphi2, dC_dtheta1, dC_dvarphi1, dC_dzeta1, dC_dzeta1c, dC_dzeta2, dC_dzeta2c, dC_dtheta, dC_dvarphi])

    # dmu
    dmu_dgamma1 = np.array([1.0, 0.0, 0.0, 0.0],dtype = np.complex128)
    dmu_dgamma1c = np.array([e_iphi1**2*(1/e_ivarphi1**2*T2*sin_theta1**2+T1*cos_theta1**2), e_iphi12*sin_theta1*cos_theta1*(e_ivarphi1*T1-1/e_ivarphi1*T2),-e_iphi1*cos_theta*cos_theta1/cosh_r1+e_iphi1*e_ivarphi/e_ivarphi1*sin_theta*sin_theta1/cosh_r2 , e_iphi1/e_ivarphi*cos_theta1*sin_theta/cosh_r1+e_iphi1/e_ivarphi1*cos_theta*sin_theta1/cosh_r2], dtype = np.complex128)
    dmu_dgamma2 = np.array([0.0, 1.0, 0.0, 0.0], dtype = np.complex128)
    dmu_dgamma2c = np.array([e_iphi12*sin_theta1*cos_theta1*(e_ivarphi1*T1-1/e_ivarphi1*T2), e_iphi2**2*(T2*cos_theta1**2+e_ivarphi1**2*T1*sin_theta1**2),-e_iphi2*(e_ivarphi*sin_theta*cos_theta1/cosh_r2 + e_ivarphi1*sin_theta1*cos_theta/cosh_r1) , e_iphi2*(-cos_theta*cos_theta1/cosh_r2 + 1/e_ivarphi*e_ivarphi1*sin_theta1*sin_theta/cosh_r1)], dtype = np.complex128)

    dmu_dphi1 = np.array([ 2j*e_iphi1**2*gamma1c*(1/e_ivarphi1**2*T2*sin_theta1**2 + T1*cos_theta1**2) + 1j*gamma2c*sin_theta1*cos_theta1*e_iphi12*(e_ivarphi1*T1-1/e_ivarphi1*T2),1j*gamma1c*sin_theta1*cos_theta1*e_iphi12*(e_ivarphi1*T1-1/e_ivarphi1*T2) , 1j*e_iphi1*gamma1c*(-cos_theta*cos_theta1/cosh_r1+e_ivarphi/e_ivarphi1*sin_theta*sin_theta1/cosh_r2),1j*e_iphi1*gamma1c*(1/e_ivarphi1*sin_theta1*cos_theta/cosh_r2+1/e_ivarphi*sin_theta*cos_theta1/cosh_r1) ],dtype = np.complex128)
    dmu_dphi2 = np.array([1j*gamma2c*sin_theta1*cos_theta1*e_iphi12*(e_ivarphi1*T1-1/e_ivarphi1*T2) , 2j*e_iphi2**2*gamma2c*(e_ivarphi1**2*T1*sin_theta1**2 + T2*cos_theta1**2) + 1j*gamma1c*sin_theta1*cos_theta1*e_iphi12*(e_ivarphi1*T1-1/e_ivarphi1*T2), -1j*e_iphi2*gamma2c*(e_ivarphi*sin_theta*cos_theta1/cosh_r2+e_ivarphi1*sin_theta1*cos_theta/cosh_r1) , 1j*e_iphi2*gamma2c*(-cos_theta*cos_theta1/cosh_r2+e_ivarphi1/e_ivarphi*sin_theta*sin_theta1/cosh_r1)],dtype = np.complex128)

    dmu_dtheta1 = np.array([(e_ivarphi1*T1-1/e_ivarphi1*T2)*(e_iphi12*gamma2c*np.cos(2*theta1)-e_iphi1**2/e_ivarphi1*gamma1c*np.sin(2*theta1)) , (e_ivarphi1*T1-1/e_ivarphi1*T2)*(e_iphi12*gamma1c*np.cos(2*theta1)+e_iphi2**2*e_ivarphi1*gamma2c*np.sin(2*theta1)), e_iphi1*gamma1c*(sin_theta1*cos_theta/cosh_r1+e_ivarphi/e_ivarphi1*sin_theta*cos_theta1/cosh_r2)+ e_iphi2*gamma2c*(e_ivarphi*sin_theta*sin_theta1/cosh_r2 - e_ivarphi1*cos_theta*cos_theta1/cosh_r1), cos_theta/cosh_r2*(e_iphi2*gamma2c*sin_theta1+e_iphi1/e_ivarphi1*gamma1c*cos_theta1)+1/e_ivarphi*sin_theta/cosh_r1*(-e_iphi1*gamma1c*sin_theta1+e_iphi2*e_ivarphi1*gamma2c*cos_theta1)],dtype = np.complex128)
    dmu_dvarphi1 = np.array([1j*sin_theta1*(e_iphi12*(e_ivarphi1*T1+1/e_ivarphi1*T2)*gamma2c*cos_theta1-2*e_iphi1**2/e_ivarphi1**2*T2*gamma1c*sin_theta1) , 1j*sin_theta1*(e_iphi12*(e_ivarphi1*T1+1/e_ivarphi1*T2)*gamma1c*cos_theta1+2*e_iphi2**2*e_ivarphi1**2*T1*gamma2c*sin_theta1), -1j*sin_theta1*(e_iphi2*e_ivarphi1*gamma2c*cos_theta/cosh_r1+e_iphi1*e_ivarphi/e_ivarphi1*gamma1c*sin_theta/cosh_r2), 1j*sin_theta1*(e_iphi2*e_ivarphi1/e_ivarphi*gamma2c*sin_theta/cosh_r1 - e_iphi1/e_ivarphi1*gamma1c*cos_theta/cosh_r2)],dtype = np.complex128)

    dmu_dr1 = np.array([ 1/cosh_r1**2*e_idelta1*(gamma1c*e_iphi1**2*cos_theta1**2+gamma2c*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1), 1/cosh_r1**2*e_idelta1*(gamma2c*e_iphi2**2*e_ivarphi1**2*sin_theta1**2+gamma1c*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1), 1/cosh_r1*tanh_r1*cos_theta*(gamma2c*e_iphi2*e_ivarphi1*sin_theta1+gamma1c*e_iphi1*cos_theta1), -1/cosh_r1*tanh_r1/e_ivarphi*sin_theta*(gamma1c*e_iphi1*cos_theta1+gamma2c*e_iphi2*e_ivarphi1*sin_theta1)], dtype = np.complex128)
    dmu_ddelta1 = np.array([ 1j*tanh_r1*e_idelta1*(gamma1c*e_iphi1**2*cos_theta1**2+gamma2c*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1), 1j*tanh_r1*e_idelta1*(gamma2c*e_iphi2**2*e_ivarphi1**2*sin_theta1**2+gamma1c*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1), 0.0, 0.0], dtype=np.complex128)
    if r1 > 0.01:
        dmu_ddelta1_over_r = dmu_ddelta1/r1
    else: # Taylor series for tanh(r)/r
        dmu_ddelta1_over_r = np.array([ (1 - r1**2/3. + 2*r1**4/15.)*1j*e_idelta1*(gamma1c*e_iphi1**2*cos_theta1**2+gamma2c*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1), 1j*(1 - r1**2/3. + 2*r1**4/15.)*e_idelta1*(gamma2c*e_iphi2**2*e_ivarphi1**2*sin_theta1**2+gamma1c*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1), 0.0, 0.0], dtype=np.complex128)
    dmu_dzeta1 = np.exp(-1j*delta1)*(dmu_dr1 - 1j*dmu_ddelta1_over_r)/2
    dmu_dzeta1c = e_idelta1*(dmu_dr1 + 1j*dmu_ddelta1_over_r)/2

    dmu_dr2 = np.array([ 1/cosh_r2**2*np.exp(1j*delta2)*(gamma1c*e_iphi1**2/e_ivarphi1**2*sin_theta1**2 - gamma2c*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1), 1/cosh_r2**2*np.exp(1j*delta2)*(gamma2c*e_iphi2**2*cos_theta1**2 - gamma1c*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1), 1/cosh_r2*tanh_r2*e_ivarphi*(gamma2c*e_iphi2*sin_theta*cos_theta1 - gamma1c*e_iphi1/e_ivarphi1*sin_theta*sin_theta1), 1/cosh_r2*tanh_r2*cos_theta*(gamma2c*e_iphi2*cos_theta1 - gamma1c*e_iphi1/e_ivarphi1*sin_theta1)], dtype=np.complex128)
    dmu_ddelta2 = np.array([ 1j*tanh_r2*np.exp(1j*delta2)*(gamma1c*e_iphi1**2/e_ivarphi1**2*sin_theta1**2 - gamma2c*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1),  1j*tanh_r2*np.exp(1j*delta2)*(gamma2c*e_iphi2**2*cos_theta1**2 - gamma1c*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1), 0.0, 0.0], dtype=np.complex128)
    if r2 > 0.01:
        dmu_ddelta2_over_r = dmu_ddelta2/r2
    else: # Taylor series for tanh(r)/r
        dmu_ddelta2_over_r = np.array([ (1 - r2**2/3. + 2*r2**4/15.)*1j*np.exp(1j*delta2)*(gamma1c*e_iphi1**2/e_ivarphi1**2*sin_theta1**2 - gamma2c*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1),  (1 - r2**2/3. + 2*r2**4/15.)*1j*np.exp(1j*delta2)*(gamma2c*e_iphi2**2*cos_theta1**2 - gamma1c*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1), 0.0, 0.0], dtype=np.complex128)
    dmu_dzeta2 = np.exp(-1j*delta2)*(dmu_dr2 - 1j*dmu_ddelta2_over_r)/2
    dmu_dzeta2c = np.exp(1j*delta2)*(dmu_dr2 + 1j*dmu_ddelta2_over_r)/2


    dmu_dtheta = np.array([0.0, 0.0, e_iphi1*gamma1c*(sin_theta*cos_theta1/cosh_r1+e_ivarphi/e_ivarphi1*cos_theta*sin_theta1/cosh_r2)+ e_iphi2*gamma2c*(-e_ivarphi*cos_theta*cos_theta1/cosh_r2+e_ivarphi1*sin_theta*sin_theta1/cosh_r1), sin_theta/cosh_r2*(e_iphi2*gamma2c*cos_theta1-e_iphi1/e_ivarphi1*gamma1c*sin_theta1)+1/e_ivarphi*cos_theta/cosh_r1*(e_iphi1*gamma1c*cos_theta1+e_iphi2*e_ivarphi1*gamma2c*sin_theta1)], dtype=np.complex128)
    dmu_dvarphi = np.array([0.0, 0.0, -1j*e_ivarphi/cosh_r2*sin_theta*(e_iphi2*gamma2c*cos_theta1-e_iphi1/e_ivarphi1*gamma1c*sin_theta1), -1j/e_ivarphi/cosh_r1*sin_theta*(e_iphi1*gamma1c*cos_theta1+e_iphi2*e_ivarphi1*gamma2c*sin_theta1)], dtype=np.complex128)


    dmu = np.stack((dmu_dgamma1, dmu_dgamma1c, dmu_dgamma2, dmu_dgamma2c, dmu_dphi1, dmu_dphi2, dmu_dtheta1, dmu_dvarphi1, dmu_dzeta1, dmu_dzeta1c, dmu_dzeta2, dmu_dzeta2c, dmu_dtheta, dmu_dvarphi), axis=-1)

    # dSigma
    dSigma_dgamma1 = np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]], dtype=np.complex128)
    dSigma_dgamma1c = np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]], dtype=np.complex128)
    dSigma_dgamma2 = np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]], dtype=np.complex128)
    dSigma_dgamma2c = np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]], dtype=np.complex128)

    dSigma1_dphi1 = np.array([[2j*e_iphi1**2/e_ivarphi1**2*(e_ivarphi1**2*T1*cos_theta1**2 + T2*sin_theta1**2), 1j*e_iphi12/e_ivarphi1*(e_ivarphi1**2*T1-T2)*sin_theta1*cos_theta1],[1j*e_iphi12/e_ivarphi1*(e_ivarphi1**2*T1-T2)*sin_theta1*cos_theta1, 0.0]], dtype=np.complex128)
    dSigma2_dphi1 = np.array([[1j*e_iphi1*(-cos_theta*cos_theta1/cosh_r1+e_ivarphi/e_ivarphi1*sin_theta1*sin_theta/cosh_r2),  1j*e_iphi1*(1/e_ivarphi*sin_theta*cos_theta1/cosh_r1 + 1/e_ivarphi1*sin_theta1*cos_theta/cosh_r2)],[0.0 + 0.0*1j, 0.0 + 0.0*1j]], dtype=np.complex128)
    dSigma3_dphi1 = dSigma2_dphi1.T
    dSigma4_dphi1 = np.array([[0.0,0.0],[0.0,0.0]], dtype=np.complex128)

    dSigma_dphi1 = np.concatenate((np.concatenate( (dSigma1_dphi1,dSigma2_dphi1) ,axis=1),np.concatenate((dSigma3_dphi1, dSigma4_dphi1),axis=1)))


    dSigma1_dphi2 = np.array([[0.0, 1j*e_iphi12/e_ivarphi1*(e_ivarphi1**2*T1-T2)*sin_theta1*cos_theta1],[1j*e_iphi12/e_ivarphi1*(e_ivarphi1**2*T1-T2)*sin_theta1*cos_theta1, 2j*e_iphi2**2*(T2*cos_theta1**2+e_ivarphi1**2*T1*sin_theta1**2)]], dtype=np.complex128)
    dSigma2_dphi2 = np.array([[0.0 + 0.0*1j, 0.0 + 0.0*1j],[-1j*e_iphi2*(e_ivarphi*sin_theta*cos_theta1/cosh_r2 + e_ivarphi1*sin_theta1*cos_theta/cosh_r1), 1j*e_iphi2*(e_ivarphi1/e_ivarphi*sin_theta*sin_theta1/cosh_r1 - cos_theta1*cos_theta/cosh_r2)]], dtype=np.complex128)
    dSigma3_dphi2 = dSigma2_dphi2.T
    dSigma4_dphi2 = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)

    dSigma_dphi2 = np.concatenate((np.concatenate( (dSigma1_dphi2,dSigma2_dphi2) ,axis=1),np.concatenate((dSigma3_dphi2, dSigma4_dphi2),axis=1)))

    dSigma1_dtheta1 = np.array([[e_iphi1**2*(-T1+1/e_ivarphi1**2*T2)*np.sin(2*theta1), e_iphi12*(e_ivarphi1*T1-1/e_ivarphi1*T2)*np.cos(2*theta1)],[e_iphi12*(e_ivarphi1*T1-1/e_ivarphi1*T2)*np.cos(2*theta1), e_iphi2**2*(e_ivarphi1**2*T1-T2)*np.sin(2*theta1)]], dtype=np.complex128)
    dSigma2_dtheta1 = np.array([[e_iphi1*(e_ivarphi/e_ivarphi1*sin_theta*cos_theta1/cosh_r2 + sin_theta1*cos_theta/cosh_r1), e_iphi1*(1/e_ivarphi1*cos_theta*cos_theta1/cosh_r2 - 1/e_ivarphi*sin_theta*sin_theta1/cosh_r1)],[e_iphi2*(-e_ivarphi1*cos_theta*cos_theta1/cosh_r1 + e_ivarphi*sin_theta1*sin_theta/cosh_r2), e_iphi2*(sin_theta1*cos_theta/cosh_r2 + e_ivarphi1/e_ivarphi*sin_theta*cos_theta1/cosh_r1)]], dtype=np.complex128)
    dSigma3_dtheta1 = dSigma2_dtheta1.T
    dSigma4_dtheta1 = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)

    dSigma_dtheta1 = np.concatenate((np.concatenate( (dSigma1_dtheta1,dSigma2_dtheta1) ,axis=1),np.concatenate((dSigma3_dtheta1, dSigma4_dtheta1),axis=1)))



    dSigma1_dvarphi1 = np.array([[-2j*e_iphi1**2/e_ivarphi1**2*T2*sin_theta1**2, 1j*e_iphi12*(e_ivarphi1*T1 + 1/e_ivarphi1*T2)*sin_theta1*cos_theta1],[1j*e_iphi12*(e_ivarphi1*T1 + 1/e_ivarphi1*T2)*sin_theta1*cos_theta1, 2j*e_iphi2**2*e_ivarphi1**2*T1*sin_theta1**2]], dtype=np.complex128)
    dSigma2_dvarphi1 = np.array([[-1j*e_iphi1*e_ivarphi/e_ivarphi1*sin_theta*sin_theta1/cosh_r2, -1j*e_iphi1/e_ivarphi1*sin_theta1*cos_theta/cosh_r2],[-1j*e_iphi2*e_ivarphi1*sin_theta1*cos_theta/cosh_r1, 1j*e_iphi2*e_ivarphi1/e_ivarphi*sin_theta*sin_theta1/cosh_r1]], dtype=np.complex128)
    dSigma3_dvarphi1 = dSigma2_dvarphi1.T
    dSigma4_dvarphi1 = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)

    dSigma_dvarphi1 = np.concatenate((np.concatenate( (dSigma1_dvarphi1,dSigma2_dvarphi1) ,axis=1),np.concatenate((dSigma3_dvarphi1, dSigma4_dvarphi1),axis=1)))

    #np.array([[],[]], dtype=dtype)
    dSigma1_dr1 = np.array([[e_idelta1*e_iphi1**2*cos_theta1**2/cosh_r1**2, e_idelta1*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1/cosh_r1**2], [e_idelta1*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1/cosh_r1**2, e_idelta1*e_iphi2**2*e_ivarphi1**2*sin_theta1**2/cosh_r1**2]], dtype=np.complex128)
    dSigma2_dr1 = np.array([[e_iphi1*cos_theta*cos_theta1/cosh_r1*tanh_r1, -e_iphi1/e_ivarphi*sin_theta*cos_theta1/cosh_r1*tanh_r1],[e_iphi2*e_ivarphi1*sin_theta1*cos_theta/cosh_r1*tanh_r1, -e_iphi2*e_ivarphi1/e_ivarphi*sin_theta*sin_theta1/cosh_r1*tanh_r1]], dtype=np.complex128)
    dSigma3_dr1 = dSigma2_dr1.T
    dSigma4_dr1 = np.array([[-np.exp(-1j*delta1)*cos_theta**2/cosh_r1**2, np.exp(-1j*delta1)/e_ivarphi*sin_theta*cos_theta/cosh_r1**2],[np.exp(-1j*delta1)/e_ivarphi*sin_theta*cos_theta/cosh_r1**2, -np.exp(-1j*delta1)/e_ivarphi**2*sin_theta**2/cosh_r1**2]], dtype=np.complex128)

    dSigma_dr1 = np.concatenate((np.concatenate( (dSigma1_dr1,dSigma2_dr1) ,axis=1),np.concatenate((dSigma3_dr1, dSigma4_dr1),axis=1)))


    dSigma1_ddelta1 = np.array([[1j*e_idelta1*e_iphi1**2*cos_theta1**2*tanh_r1, 1j*e_idelta1*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1*tanh_r1], [1j*e_idelta1*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1*tanh_r1, 1j*e_idelta1*e_iphi2**2*e_ivarphi1**2*sin_theta1**2*tanh_r1]], dtype=np.complex128)
    dSigma2_ddelta1 = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)
    dSigma3_ddelta1 = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)
    dSigma4_ddelta1 = np.array([[1j*np.exp(-1j*delta1)*cos_theta**2*tanh_r1, -1j*np.exp(-1j*delta1)/e_ivarphi*sin_theta*cos_theta*tanh_r1],[-1j*np.exp(-1j*delta1)/e_ivarphi*sin_theta*cos_theta*tanh_r1, 1j*np.exp(-1j*delta1)/e_ivarphi**2*sin_theta**2*tanh_r1]], dtype=np.complex128)

    dSigma_ddelta1 = np.concatenate((np.concatenate( (dSigma1_ddelta1,dSigma2_ddelta1) ,axis=1),np.concatenate((dSigma3_ddelta1, dSigma4_ddelta1),axis=1)))

    if r1 > 0.01:
        dSigma_ddelta1_over_r = dSigma_ddelta1/r1
    else: # Taylor series for tanh(r)/r = (1 - r**2/3. + 2*r**4/15.)
        dSigma1_ddelta1_over_r = np.array([[1j*e_idelta1*e_iphi1**2*cos_theta1**2*(1 - r1**2/3. + 2*r1**4/15.), 1j*e_idelta1*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1*(1 - r1**2/3. + 2*r1**4/15.)], [1j*e_idelta1*e_iphi12*e_ivarphi1*sin_theta1*cos_theta1*(1 - r1**2/3. + 2*r1**4/15.), 1j*e_idelta1*e_iphi2**2*e_ivarphi1**2*sin_theta1**2*(1 - r1**2/3. + 2*r1**4/15.)]], dtype=np.complex128)
        dSigma2_ddelta1_over_r = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)
        dSigma3_ddelta1_over_r = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)
        dSigma4_ddelta1_over_r = np.array([[1j*np.exp(-1j*delta1)*cos_theta**2*(1 - r1**2/3. + 2*r1**4/15.), -1j*np.exp(-1j*delta1)/e_ivarphi*sin_theta*cos_theta*(1 - r1**2/3. + 2*r1**4/15.)],[-1j*np.exp(-1j*delta1)/e_ivarphi*sin_theta*cos_theta*(1 - r1**2/3. + 2*r1**4/15.), 1j*np.exp(-1j*delta1)/e_ivarphi**2*sin_theta**2*(1 - r1**2/3. + 2*r1**4/15.)]], dtype=np.complex128)

        dSigma_ddelta1_over_r = np.concatenate((np.concatenate( (dSigma1_ddelta1_over_r,dSigma2_ddelta1_over_r) ,axis=1),np.concatenate((dSigma3_ddelta1_over_r, dSigma4_ddelta1_over_r),axis=1)))

    dSigma_dzeta1 = np.exp(-1j*delta1)*(dSigma_dr1 - 1j*dSigma_ddelta1_over_r)/2
    dSigma_dzeta1c = e_idelta1*(dSigma_dr1 + 1j*dSigma_ddelta1_over_r)/2


    dSigma1_dr2 = np.array([[np.exp(1j*delta2)*e_iphi1**2/e_ivarphi1**2*sin_theta1**2/cosh_r2**2, -np.exp(1j*delta2)*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1/cosh_r2**2], [-np.exp(1j*delta2)*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1/cosh_r2**2, np.exp(1j*delta2)*e_iphi2**2*cos_theta1**2/cosh_r2**2]], dtype=np.complex128)
    dSigma2_dr2 = np.array([[-e_iphi1*e_ivarphi/e_ivarphi1*sin_theta*sin_theta1/cosh_r2*tanh_r2, -e_iphi1/e_ivarphi1*sin_theta1*cos_theta/cosh_r2*tanh_r2],[e_iphi2*e_ivarphi*sin_theta*cos_theta1/cosh_r2*tanh_r2, e_iphi2*cos_theta*cos_theta1/cosh_r2*tanh_r2]], dtype=np.complex128)
    dSigma3_dr2 = dSigma2_dr2.T
    dSigma4_dr2 = np.array([[-np.exp(-1j*delta2)*e_ivarphi**2*sin_theta**2/cosh_r2**2, -np.exp(-1j*delta2)*e_ivarphi*sin_theta*cos_theta/cosh_r2**2],[-np.exp(-1j*delta2)*e_ivarphi*sin_theta*cos_theta/cosh_r2**2, -np.exp(-1j*delta2)*cos_theta**2/cosh_r2**2]], dtype=np.complex128)

    dSigma_dr2 = np.concatenate((np.concatenate( (dSigma1_dr2,dSigma2_dr2) ,axis=1),np.concatenate((dSigma3_dr2, dSigma4_dr2),axis=1)))


    dSigma1_ddelta2 = np.array([[1j*np.exp(1j*delta2)*e_iphi1**2/e_ivarphi1**2*sin_theta1**2*tanh_r2, -1j*np.exp(1j*delta2)*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1*tanh_r2], [-1j*np.exp(1j*delta2)*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1*tanh_r2, 1j*np.exp(1j*delta2)*e_iphi2**2*cos_theta1**2*tanh_r2]], dtype=np.complex128)
    dSigma2_ddelta2 = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)
    dSigma3_ddelta2 = -dSigma2_ddelta2.T
    dSigma4_ddelta2 = np.array([[1j*np.exp(-1j*delta2)*e_ivarphi**2*sin_theta**2*tanh_r2, 1j*np.exp(-1j*delta2)*e_ivarphi*sin_theta*cos_theta*tanh_r2],[1j*np.exp(-1j*delta2)*e_ivarphi*sin_theta*cos_theta*tanh_r2, 1j*np.exp(-1j*delta2)*cos_theta**2*tanh_r2]], dtype=np.complex128)

    dSigma_ddelta2 = np.concatenate((np.concatenate( (dSigma1_ddelta2,dSigma2_ddelta2) ,axis=1),np.concatenate((dSigma3_ddelta2, dSigma4_ddelta2),axis=1)))

    if r2 > 0.01:
        dSigma_ddelta2_over_r = dSigma_ddelta2/r2
    else: # Taylor series for tanh(r)/r = (1 - r**2/3. + 2*r**4/15.)
        dSigma1_ddelta2_over_r = np.array([[1j*np.exp(1j*delta2)*e_iphi1**2/e_ivarphi1**2*sin_theta1**2*(1 - r2**2/3. + 2*r2**4/15.), -1j*np.exp(1j*delta2)*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1*(1 - r2**2/3. + 2*r2**4/15.)], [-1j*np.exp(1j*delta2)*e_iphi12/e_ivarphi1*sin_theta1*cos_theta1*(1 - r2**2/3. + 2*r2**4/15.), 1j*np.exp(1j*delta2)*e_iphi2**2*cos_theta1**2*(1 - r2**2/3. + 2*r2**4/15.)]], dtype = np.complex128)
        dSigma2_ddelta2_over_r = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)
        dSigma3_ddelta2_over_r = np.array([[0.0, 0.0],[ 0.0, 0.0]], dtype=np.complex128)
        dSigma4_ddelta2_over_r = np.array([[1j*np.exp(-1j*delta2)*e_ivarphi**2*sin_theta**2*(1 - r2**2/3. + 2*r2**4/15.), 1j*np.exp(-1j*delta2)*e_ivarphi*sin_theta*cos_theta*(1 - r2**2/3. + 2*r2**4/15.)],[1j*np.exp(-1j*delta2)*e_ivarphi*sin_theta*cos_theta*(1 - r2**2/3. + 2*r2**4/15.), 1j*np.exp(-1j*delta2)*cos_theta**2*(1 - r2**2/3. + 2*r2**4/15.)]], dtype=np.complex128)

        dSigma_ddelta2_over_r = np.concatenate((np.concatenate( (dSigma1_ddelta2_over_r,dSigma2_ddelta2_over_r) ,axis=1),np.concatenate((dSigma3_ddelta2_over_r, dSigma4_ddelta2_over_r),axis=1)))

    dSigma_dzeta2 = np.exp(-1j*delta2)*(dSigma_dr2 - 1j*dSigma_ddelta2_over_r)/2
    dSigma_dzeta2c = np.exp(1j*delta2)*(dSigma_dr2 + 1j*dSigma_ddelta2_over_r)/2


    dSigma1_dtheta = np.array([[0.0,0.0],[0.0,0.0]], dtype=np.complex128)
    dSigma2_dtheta = np.array([[e_iphi1*(sin_theta*cos_theta1/cosh_r1 + e_ivarphi/e_ivarphi1*sin_theta1*cos_theta/cosh_r2), e_iphi1*(1/e_ivarphi*cos_theta*cos_theta1/cosh_r1 - 1/e_ivarphi1*sin_theta1*sin_theta/cosh_r2)],[e_iphi2*(-e_ivarphi*cos_theta*cos_theta1/cosh_r2 + e_ivarphi1*sin_theta1*sin_theta/cosh_r1), e_iphi2*(sin_theta*cos_theta1/cosh_r2 + e_ivarphi1/e_ivarphi*sin_theta1*cos_theta/cosh_r1)]], dtype=np.complex128)
    dSigma3_dtheta = dSigma2_dtheta.T
    dSigma4_dtheta = np.array([[np.sin(2*theta)*(T1minus-e_ivarphi**2*T2minus),1/e_ivarphi*np.cos(2*theta)*(T1minus-e_ivarphi**2*T2minus)],[1/e_ivarphi*np.cos(2*theta)*(T1minus-e_ivarphi**2*T2minus), np.sin(2*theta)*(-1/e_ivarphi**2*T1minus+T2minus)]], dtype=np.complex128)

    dSigma_dtheta = np.concatenate((np.concatenate( (dSigma1_dtheta,dSigma2_dtheta) ,axis=1),np.concatenate((dSigma3_dtheta, dSigma4_dtheta),axis=1)))

    #np.array([[],[]], dtype=dtype)
    dSigma1_dvarphi = np.array([[0.0, 0.0],[0.0, 0.0]], dtype=np.complex128)
    dSigma2_dvarphi = np.array([[1j*e_iphi1*e_ivarphi/e_ivarphi1*sin_theta*sin_theta1/cosh_r2, -1j*e_iphi1/e_ivarphi*sin_theta*cos_theta1/cosh_r1],[-1j*e_iphi2*e_ivarphi*sin_theta*cos_theta1/cosh_r2, -1j*e_iphi2*e_ivarphi1/e_ivarphi*sin_theta*sin_theta1/cosh_r1]], dtype=np.complex128)
    dSigma3_dvarphi = dSigma2_dvarphi.T
    dSigma4_dvarphi = np.array([[-2j*e_ivarphi**2*sin_theta**2*T2minus,-1j/e_ivarphi*sin_theta*cos_theta*(T1minus+e_ivarphi**2*T2minus)],[-1j/e_ivarphi*sin_theta*cos_theta*(T1minus+e_ivarphi**2*T2minus), 2j/e_ivarphi**2*sin_theta**2*T1minus]], dtype=np.complex128)

    dSigma_dvarphi = np.concatenate((np.concatenate( (dSigma1_dvarphi,dSigma2_dvarphi) ,axis=1),np.concatenate((dSigma3_dvarphi, dSigma4_dvarphi),axis=1)))

    dSigma = np.stack((dSigma_dgamma1, dSigma_dgamma1c, dSigma_dgamma2, dSigma_dgamma2c, dSigma_dphi1, dSigma_dphi2, dSigma_dtheta1, dSigma_dvarphi1, dSigma_dzeta1, dSigma_dzeta1c, dSigma_dzeta2, dSigma_dzeta2c, dSigma_dtheta, dSigma_dvarphi), axis=-1)

    return dC, dmu, dSigma


@njit
def R_matrix(gamma: np.complex, phi: np.float, z: np.complex, cutoff: int, old_state: np.array) -> np.array:
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (complex array[D]): State to be transformed

    Returns:
        R (complex array[D,D]): the matrix whose 1st column is the transformed state
    """
    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)
    cutoff = convert_scalar(cutoff)

    dtype = old_state.dtype
    # print(dtype)
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))

    R = np.zeros((cutoff, cutoff), dtype=dtype)
    G0 = np.zeros(cutoff, dtype=dtype)
    
    print("here inside R_matrix!!!!")

    # first row of Transformation matrix
    G0[0] = C
    for n in range(1, cutoff):
        G0[n] = mu[1] / sqrt[n] * G0[n - 1] - Sigma[1, 1] * sqrt[n - 1] / sqrt[n] * G0[n - 2]
    
    print(G0)
    # first row of R matrix
    for n in range(cutoff):
        R[0, n] = np.dot(G0[: cutoff - n], old_state)
        old_state = old_state[1:] * sqrt[1 : cutoff - n]

    # rest of R matrix
    for m in range(1, cutoff):
        for n in range(cutoff - m):
            R[m, n] = (
                mu[0] / sqrt[m] * R[m - 1, n]
                - Sigma[0, 0] * sqrt[m - 1] / sqrt[m] * R[m - 2, n]
                - Sigma[0, 1] / sqrt[m] * R[m - 1, n + 1]
            )

    return R

@njit
def large_squeezing(gamma: np.complex, phi: np.float, z: np.complex, cutoff: int, old_state: np.array) -> np.array:
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (complex array[D]): State to be transformed

    Returns:
        R (complex array[D,D]): the matrix whose 1st column is the transformed state
    """
    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)
    cutoff = convert_scalar(cutoff)

    dtype = old_state.dtype
    
    r = np.abs(z)
    delta = np.angle(z)
    e_idelta = np.exp(1j*delta)
    tanhr = np.tanh(r)
    sechr = 1/np.cosh(r)

    C = np.sqrt(1/np.cosh(r))
#    mu = np.array([0,0], dtype=dtype)
    Sigma = np.array([[e_idelta * tanhr, -sechr],[-sechr, -1/e_idelta * tanhr]],dtype=dtype)

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    
    R = np.zeros((cutoff, cutoff), dtype=dtype)
    G0 = np.zeros(cutoff, dtype=dtype)

    # first row of Transformation matrix
    G0[0] = C
    for n in range(1, cutoff):
        G0[n] = - Sigma[1, 1] * sqrt[n - 1] / sqrt[n] * G0[n - 2]

    R[0,0] = np.dot(G0,old_state)
    
    for m in range(2,cutoff,2):
        R[m,0] = - Sigma[0, 0] * sqrt[m - 1] / sqrt[m] * R[m - 2, 0]

    return R



@njit
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
        
        old_state(np.array(complex)): State to be transformed

    Returns:
        R (complex array[D,D,D,D]): the matrix where R[:,:,0,0] is the transformed state
    """
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

    cutoff = old_state.shape[0]
    dtype = old_state.dtype

    sqrt = np.sqrt(np.arange(cutoff, dtype = dtype))
    sqrtT = sqrt.reshape(-1, 1)


    R = np.zeros((cutoff, cutoff, cutoff+1, cutoff+1), dtype = dtype)
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
            R[0,0,j,k] = np.sum(G_00pq3*old_state[j:,k:])
            G_00pq3 = G_00pq3[:,:-1]*sqrt[k+1:]
        G_00pq2 = sqrtT[j+1:]*G_00pq2[:-1,:]

    #R_0n^jk
    for n in range(1,cutoff):
        for k in range(0,cutoff):
            for j in range(0,cutoff):
                R[0,n,j,k] = mu[1]/sqrt[n]*R[0,n-1,j,k] - Sigma[1,1]/sqrt[n]*sqrt[n-1]*R[0,n-2,j,k] - Sigma[1,2]/sqrt[n]*R[0,n-1,j+1,k] - Sigma[1,3]/sqrt[n]*R[0,n-1,j,k+1]

    #R_mn^jk
    for m in range(1,cutoff):
        for n in range(0,cutoff):
            for j in range(0,cutoff-m):
                for k in range(0,cutoff-m-j):
                    R[m,n,j,k] = mu[0]/sqrt[m]*R[m-1,n,j,k] - Sigma[0,0]/sqrt[m]*sqrt[m-1]*R[m-2,n,j,k] - Sigma[0,1]*sqrt[n]/sqrt[m]*R[m-1,n-1,j,k] - Sigma[0,2]/sqrt[m]*R[m-1,n,j+1,k] - Sigma[0,3]/sqrt[m]*R[m-1,n,j,k+1]
          
    return R



@njit
def G_matrix(
    gamma: np.complex, phi: np.float, z: np.complex, cutoff: np.int, dtype: np.dtype = np.complex128
) -> np.array:
    """
    Constructs the Gaussian transformation recursively

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        zeta (complex): squeezing parameter
        cutoff (int): Fock space cutoff dimension
        dtype (numpy dtype): dtype of the output

    Returns:
        G (complex array[cutoff]): the single-mode Gaussian transformation matrix
    """
    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)
    cutoff = convert_scalar(cutoff)

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    G = np.zeros((cutoff, cutoff), dtype=dtype)  # maybe numba cannot create array of zeros of type complex64
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)

    # First column
    G[0, 0] = C
    for m in range(cutoff - 1):
        G[m + 1, 0] = mu[0] / sqrt[m + 1] * G[m, 0] - Sigma[0, 0] * sqrt[m] / sqrt[m + 1] * G[m - 1, 0]

    # All rows
    for m in range(cutoff):
        for n in range(cutoff - 1):
            G[m, n + 1] = (
                mu[1] / sqrt[n + 1] * G[m, n]
                - Sigma[1, 0] * sqrt[m] / sqrt[n + 1] * G[m - 1, n]
                - Sigma[1, 1] * sqrt[n] / sqrt[n + 1] * G[m, n - 1]
            )

    return G

@njit
def G_matrix2(gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, cutoff, dtype = np.complex128):
    """
    Directly constructs the transformation G matrix recursively and exactly.

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

    Returns:
        (np.array(complex)): the transformation matrix G
    """
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

    G = np.zeros((cutoff, cutoff, cutoff, cutoff),dtype = dtype)

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



# Extras
@njit
def approx_new_state(gamma, phi, z, old_state, order=None):
    """
    Constructs an approximation of the transformed state by ignoring the 
    squeezing contribution after a certain order.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (np.array(complex)): State to be transformed
        order (int): order of the approximation

    Returns:
        (np.array(complex)): the new state which is exact up to dimension `order`

    """
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)

    cutoff = old_state.shape[0]
    dtype = old_state.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    if order is None:
        order = cutoff

    R = np.zeros((cutoff, order), dtype=dtype)
    G0 = np.zeros(cutoff, dtype=dtype)

    # first row of Transformation matrix
    G0[0] = C
    for n in range(1, cutoff):
        G0[n] = mu[1] / sqrt[n] * G0[n - 1] - Sigma[1, 1] * sqrt[n - 1] / sqrt[n] * G0[n - 2]

    # first row of R matrix
    for n in range(order):
        R[0, n] = np.dot(G0[: cutoff - n], old_state)
        old_state = old_state[1:] * sqrt[1 : cutoff - n]

    # rest of R matrix
    for m in range(1, cutoff):
        for n in range(min(order - 1, cutoff - m)):
            R[m, n] = (
                mu[0] / sqrt[m] * R[m - 1, n]
                - Sigma[0, 0] * sqrt[m - 1] / sqrt[m] * R[m - 2, n]
                - Sigma[0, 1] / sqrt[m] * R[m - 1, n + 1]
            )

    return R[:, 0]
