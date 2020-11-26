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

import tensorflow as tf
import numpy as np

from .jitted import G_matrix, G_matrix2,dPsi, R_matrix, dPsi2, R_matrix2, inverse_metric_real


def complex_initializer(dtype):
    f = tf.random_normal_initializer()

    def initializer(*args, dtype, **kwargs):
        real = f(*args, **kwargs)
        imag = f(*args, **kwargs)
        return tf.cast(tf.complex(real, imag), dtype)

    return initializer


def real_initializer(dtype):
    f = tf.random_normal_initializer()

    def initializer(*args, dtype, **kwargs):
        real = f(*args, **kwargs)
        return tf.cast(real, dtype)

    return initializer


def real_complex_types(dtype: tf.dtypes.DType):
    if dtype == tf.complex128:
        realtype = tf.float64
        complextype = tf.complex128
    elif dtype == tf.complex64:
        realtype = tf.float32
        complextype = tf.complex64
    else:
        raise ValueError(f"dtype can be only tf.complex128 or tf.complex64, not {dtype}")
    return realtype, complextype


@tf.custom_gradient
def LayerTransformation(gamma: tf.Variable, phi: tf.Variable, z: tf.Variable, kappa: tf.Variable, state_in: tf.Tensor) -> tf.Tensor:
    """
    Evolution of a single-mode quantum state through a Gaussian transformation parametrized by the three parameters gamma, phi and z
        and a Non-Gaussian transformation parametrized by kappa.
    This is a differentiable function that can be safely used in a TensorFlow computation.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation
        z (complex): squeezing parameter
        kappa (complex): Kerr parameter
        state_in (complex array[D]): input state of dimension D
    
    Returns:
        state_out (complex array[D]): output state of dimension D
    """
    # print(f"pre: {gamma}\n {phi}\n {z}\n {state_in}\n")
    gamma = tf.convert_to_tensor(gamma)
    phi = tf.convert_to_tensor(phi)
    z = tf.convert_to_tensor(z)
    kappa = tf.convert_to_tensor(kappa)
    state_in = tf.convert_to_tensor(state_in)
    cutoff = state_in.shape[1]

    dtype_c = state_in.dtype
    dtype_r = phi.dtype

    R = tf.numpy_function(R_matrix, [gamma, phi, z, state_in], dtype_c)
    gaussian_output = R[..., 0]
    
    Kerr = tf.exp(1j * tf.cast(kappa, dtype=dtype_c) * np.arange(cutoff) ** 2)
    state_out = Kerr * gaussian_output

    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma, phi, z, kappa, Psi)"
        G = tf.numpy_function(G_matrix, [gamma, phi, z, cutoff], dtype_c)
        dPsiG_dgamma, dPsiG_dgammac, dPsiG_dphi, dPsiG_dz, dPsiG_dzc = tf.numpy_function(
            dPsi, [gamma, phi, z, state_in, G[0], R], (dtype_c, dtype_c, dtype_c, dtype_c, dtype_c)
        )
        #dPsi_dgamma (batch,D,5)
        dPsi_dgamma = Kerr * dPsiG_dgamma
        dPsi_dgammac = Kerr * dPsiG_dgammac
        dPsi_dphi = Kerr * dPsiG_dphi
        dPsi_dz = Kerr * dPsiG_dz
        dPsi_dzc = Kerr * dPsiG_dzc
        dPsi_dkappa = 1j * np.arange(cutoff) ** 2 * state_out
        
        ##(batch,5) for G[0,0]
        ##dPsi_dgamma (batch,D,5)
        ##state_out D
        
        grad_gammac = tf.reduce_sum(dy * tf.math.conj(dPsi_dgamma) + tf.math.conj(dy) * dPsi_dgammac)
        grad_phi = 2 * tf.math.real(tf.reduce_sum(dy * tf.math.conj(dPsi_dphi)))
        grad_zc = tf.reduce_sum(dy * tf.math.conj(dPsi_dz) + tf.math.conj(dy) * dPsi_dzc)
        grad_kappa = 2 * tf.math.real(tf.reduce_sum(dy * tf.math.conj(dPsi_dkappa)))
        
        
        grad_Psic = tf.linalg.matvec(Kerr[:,None]*G, dy, adjoint_a=True) # mat-vec mult on last index of both
#        grad_Psic = tf.einsum("a,ab,ca->cb",  tf.math.conj(Kerr), tf.math.conj(G) , dy)
        ##TODO: G_C matrix [D,D,batch,5] batch = 0
        
#        dPsi_dtheta = tf.convert_to_tensor([dPsi_dgamma[0], dPsi_dgammac[0], dPsi_dz[0], dPsi_dzc[0], c*dPsi_dphi[0], c*dPsi_dphi[0], c*dPsi_dkappa[0], c*dPsi_dkappa[0]])
#        dPsi_dthetac = tf.convert_to_tensor([dPsi_dgammac[0], dPsi_dgamma[0], dPsi_dzc[0], dPsi_dz[0], c*dPsi_dphi[0], c*dPsi_dphi[0], c*dPsi_dkappa[0], c*dPsi_dkappa[0]])

        dPsi_dgamma_real = dPsi_dgamma[0]+dPsi_dgammac[0]
        dPsi_dgamma_imag = 1j*dPsi_dgammac[0] - 1j*dPsi_dgammac[0]
        dPsi_dzeta_real = dPsi_dz[0]+dPsi_dzc[0]
        dPsi_dzeta_imag = 1j*dPsi_dz[0] - 1j*dPsi_dzc[0]
        dPsi_dphi_real = dPsi_dphi[0]
        dPsi_dphi_imag = 0*dPsi_dphi[0] #1j*dPsi_dphi[0]
        dPsi_dkappa_real = dPsi_dkappa[0]
        dPsi_dkappa_imag = 0*dPsi_dkappa[0] #1j*dPsi_dkappa[0]

        
        dPsi_dthetaReal = tf.convert_to_tensor([dPsi_dgamma_real, dPsi_dgamma_imag, dPsi_dzeta_real, dPsi_dzeta_imag, dPsi_dphi_real, dPsi_dphi_imag, dPsi_dkappa_real, dPsi_dkappa_imag])


        invMetric = tf.numpy_function(inverse_metric_real, [dPsi_dthetaReal, state_out[0]], dtype_c)
        
        grad_gammac_real = tf.math.real(grad_gammac)
        grad_gammac_imag = tf.math.imag(grad_gammac)
        grad_zc_real = tf.math.real(grad_zc)
        grad_zc_imag = tf.math.imag(grad_zc)
        grad_phi_real = grad_phi
        grad_phi_imag = 0
        grad_kappa_real = grad_kappa
        grad_kappa_imag = 0
        
        updates = tf.convert_to_tensor([grad_gammac_real, grad_gammac_imag, grad_zc_real, grad_zc_imag, grad_phi_real, grad_phi_imag, grad_kappa_real, grad_kappa_imag], dtype=dtype_c)
        NG_updates = tf.linalg.matvec(invMetric, updates)
        grad_gammac, grad_zc, grad_phi, grad_kappa = NG_updates[0]+1j*NG_updates[1], NG_updates[2]+1j*NG_updates[3], tf.math.real(NG_updates[4]+1j*NG_updates[5]), tf.math.real(NG_updates[6]+1j*NG_updates[7])


        return grad_gammac, grad_phi, grad_zc, grad_kappa, grad_Psic

    return state_out, grad
    
    
@tf.custom_gradient
def LayerTransformation2mode(gamma1: tf.Variable, gamma2: tf.Variable, phi1: tf.Variable, phi2: tf.Variable, theta1: tf.Variable, varphi1: tf.Variable, zeta1: tf.Variable, zeta2: tf.Variable, theta: tf.Variable, varphi: tf.Variable, kappa1: tf.Variable, kappa2: tf.Variable, state_in: tf.Tensor, dtype = np.complex128) -> tf.Tensor:
    """
    Evolution of a two-mode quantum state through a Gaussian transformation parametrized by the ten parameters gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi
        and a Non-Gaussian transformation parametrized by kappa1 and kappa2.
    This is a differentiable function that can be safely used in a TensorFlow computation.

    Arguments:
        gamma1 (complex): displacement parameter on the first mode
        gamma2 (complex): displacement parameter on the second mode
        phi1 (float): phase rotation on the first mode
        phi2 (float): phase rotation on the second mode
        theta1 (float) : Beamsplitter1 parameter
        varphi1 (float) : Beamsplitter1 parameter
        zeta1 (complex): squeezing parameter on the first mode
        zeta2 (complex): squeezing parameter on the second mode
        theta (float) : Beamsplitter2 parameter
        varphi (float) : Beamsplitter2 parameter
        kappa1 (complex): Kerr parameter on the first mode
        kappa2 (complex): Kerr parameter on the second mode
        state_in (complex array[D,D]): input state of dimension D*D
    
    Returns:
        state_out (complex array[D*D]): output state of dimension D*D
    """
    gamma1 = tf.convert_to_tensor(gamma1)
    gamma2 = tf.convert_to_tensor(gamma2)
    phi1 = tf.convert_to_tensor(phi1)
    phi2 = tf.convert_to_tensor(phi2)
    theta1 = tf.convert_to_tensor(theta1)
    varphi1 = tf.convert_to_tensor(varphi1)
    zeta1 = tf.convert_to_tensor(zeta1)
    zeta2 = tf.convert_to_tensor(zeta2)
    theta = tf.convert_to_tensor(theta)
    varphi = tf.convert_to_tensor(varphi)
    kappa1 = tf.convert_to_tensor(kappa1)
    kappa2 = tf.convert_to_tensor(kappa2)
    state_in = tf.convert_to_tensor(state_in)
    cutoff = state_in.shape[1]
    
    dtype_c = state_in.dtype
    dtype_r = phi1.dtype
    
    R = tf.numpy_function(R_matrix2, [gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, state_in], dtype_c)
    gaussian_output = R[:, :, :, 0, 0]
    Kerr1 = tf.exp(1j * tf.cast(kappa1, dtype=dtype_c) * np.arange(cutoff) ** 2)
    Kerr2 = tf.exp(1j * tf.cast(kappa2, dtype=dtype_c) * np.arange(cutoff) ** 2)
    
    state_out = Kerr1[:,None] * gaussian_output * Kerr2[None,:]

    
    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, state_in)"
        G = tf.numpy_function(G_matrix2, [gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, cutoff], dtype_c)
        
        dPsiG_dgamma1, dPsiG_dgamma1c, dPsiG_dgamma2, dPsiG_dgamma2c, dPsiG_dphi1, dPsiG_dphi2, dPsiG_dtheta1, dPsiG_dvarphi1, dPsiG_dzeta1, dPsiG_dzeta1c, dPsiG_dzeta2, dPsiG_dzeta2c, dPsiG_dtheta, dPsiG_dvarphi = tf.numpy_function(dPsi2,[gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, state_in, G[0,0,:,:], R], (dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c))
        #dPsi_dgamma (batch,D,D,14)
        dPsi_dgamma1 = Kerr1[:,None] * dPsiG_dgamma1 * Kerr2[None,:]
        dPsi_dgamma1c = Kerr1[:,None] * dPsiG_dgamma1c * Kerr2[None,:]
        dPsi_dgamma2 = Kerr1[:,None] * dPsiG_dgamma2 * Kerr2[None,:]
        dPsi_dgamma2c = Kerr1[:,None] * dPsiG_dgamma2c * Kerr2[None,:]
        dPsi_dphi1 = Kerr1[:,None] * dPsiG_dphi1 * Kerr2[None,:]
        dPsi_dphi2 = Kerr1[:,None] * dPsiG_dphi2 * Kerr2[None,:]
        dPsi_dtheta1 = Kerr1[:,None] * dPsiG_dtheta1 * Kerr2[None,:]
        dPsi_dvarphi1 = Kerr1[:,None] * dPsiG_dvarphi1 * Kerr2[None,:]
        dPsi_dzeta1 = Kerr1[:,None] * dPsiG_dzeta1 * Kerr2[None,:]
        dPsi_dzeta1c = Kerr1[:,None] * dPsiG_dzeta1c * Kerr2[None,:]
        dPsi_dzeta2 = Kerr1[:,None] * dPsiG_dzeta2 * Kerr2[None,:]
        dPsi_dzeta2c = Kerr1[:,None] * dPsiG_dzeta2c * Kerr2[None,:]
        dPsi_dtheta = Kerr1[:,None] * dPsiG_dtheta * Kerr2[None,:]
        dPsi_dvarphi = Kerr1[:,None] * dPsiG_dvarphi * Kerr2[None,:]
        
        dPsi_dkappa1 = (1j * np.arange(cutoff) ** 2)[:,None] * state_out
        dPsi_dkappa2 =  state_out * (1j * np.arange(cutoff) ** 2)[None,:]
        
        ##TODO: G_C matrix
        
        grad_gamma1c = tf.reduce_sum(dy*tf.math.conj(dPsi_dgamma1) + tf.math.conj(dy)*dPsi_dgamma1c)
        grad_gamma2c = tf.reduce_sum(dy*tf.math.conj(dPsi_dgamma2) + tf.math.conj(dy)*dPsi_dgamma2c)
        grad_zeta1c = tf.reduce_sum(dy*tf.math.conj(dPsi_dzeta1) + tf.math.conj(dy)*dPsi_dzeta1c)
        grad_zeta2c = tf.reduce_sum(dy*tf.math.conj(dPsi_dzeta2) + tf.math.conj(dy)*dPsi_dzeta2c)
        grad_phi1 = 2*tf.math.real(tf.reduce_sum(dy*tf.math.conj(dPsi_dphi1)))
        grad_phi2 = 2*tf.math.real(tf.reduce_sum(dy*tf.math.conj(dPsi_dphi2)))
        grad_theta1 = 2*tf.math.real(tf.reduce_sum(dy*tf.math.conj(dPsi_dtheta1)))
        grad_varphi1 = 2*tf.math.real(tf.reduce_sum(dy*tf.math.conj(dPsi_dvarphi1)))
        grad_theta = 2*tf.math.real(tf.reduce_sum(dy*tf.math.conj(dPsi_dtheta)))
        grad_varphi = 2*tf.math.real(tf.reduce_sum(dy*tf.math.conj(dPsi_dvarphi)))
        
        grad_k1 = 2 * tf.math.real(tf.reduce_sum(dy * tf.math.conj(dPsi_dkappa1)))
        grad_k2 = 2 * tf.math.real(tf.reduce_sum(dy * tf.math.conj(dPsi_dkappa2)))
        
        grad_Psic = tf.einsum("a,abcd,b,eab->ecd",  tf.math.conj(Kerr1), tf.math.conj(G) , tf.math.conj(Kerr2), dy)
        
        return grad_gamma1c, grad_gamma2c, grad_phi1, grad_phi2, grad_theta1, grad_varphi1, grad_zeta1c, grad_zeta2c, grad_theta, grad_varphi, grad_k1, grad_k2, grad_Psic
    
    return state_out, grad
#
#def KerrDiagonal(k: tf.Variable, cutoff: int, dtype: tf.dtypes.DType):
#    """
#    Returns the diagonal of the single-mode Kerr matrix (vector)
#
#    Arguments:
#        cutoff (int): the cutoff dimension of Fock space
#        dtype (tf dtype): either tf.complex64 or tf.complex128
#    """
#    return tf.exp(1j * tf.cast(k, dtype=dtype) * np.arange(cutoff) ** 2)
#
#
#
