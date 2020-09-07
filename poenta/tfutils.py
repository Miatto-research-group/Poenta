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

from .jitted import G_matrix, G_matrix2
from .experimental import dPsi, R_matrix, dPsi2, R_matrix2


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
def GaussianTransformation(gamma: tf.Variable, phi: tf.Variable, z: tf.Variable, state_in: tf.Tensor) -> tf.Tensor:
    """
    Evolution of a single-mode quantum state through a Gaussian transformation parametrized
    by the three parameters gamma, phi and z.
    This is a differentiable function that can be safely used in a TensorFlow computation.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation
        z (complex): squeezing parameter
        state_in (complex array[D]): input state of dimension D
    
    Returns:
        state_out (complex array[D]): output state of dimension D
    """
    # print(f"pre: {gamma}\n {phi}\n {z}\n {state_in}\n")
    gamma = tf.convert_to_tensor(gamma)
    phi = tf.convert_to_tensor(phi)
    z = tf.convert_to_tensor(z)
    state_in = tf.convert_to_tensor(state_in)
    cutoff = state_in.shape[1]

    dtype_c = state_in.dtype
    dtype_r = phi.dtype

    R = tf.numpy_function(R_matrix, [gamma, phi, z, state_in], dtype_c)
    state_out = R[..., 0]

    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma, phi, z, Psi)"
        G = tf.numpy_function(G_matrix, [gamma, phi, z, cutoff], dtype_c)
        dPsi_dgamma, dPsi_dgammac, dPsi_dphi, dPsi_dz, dPsi_dzc = tf.numpy_function(
            dPsi, [gamma, phi, z, state_in, G[0], R], (dtype_c, dtype_c, dtype_c, dtype_c, dtype_c)
        )
        grad_gammac = tf.reduce_sum(dy * tf.math.conj(dPsi_dgamma) + tf.math.conj(dy) * dPsi_dgammac)
        grad_phi = 2 * tf.math.real(tf.reduce_sum(dy * tf.math.conj(dPsi_dphi)))
        grad_zc = tf.reduce_sum(dy * tf.math.conj(dPsi_dz) + tf.math.conj(dy) * dPsi_dzc)
        grad_Psic = tf.linalg.matvec(G, dy, adjoint_a=True) # mat-vec mult on last index of both
        return grad_gammac, grad_phi, grad_zc, grad_Psic

    return state_out, grad
    
    
@tf.custom_gradient
def GaussianTransformation2mode(gamma1: tf.Variable, gamma2: tf.Variable, phi1: tf.Variable, phi2: tf.Variable, theta1: tf.Variable, varphi1: tf.Variable, zeta1: tf.Variable, zeta2: tf.Variable, theta: tf.Variable, varphi: tf.Variable, state_in: tf.Tensor, dtype = np.complex128) -> tf.Tensor:
    """
    Direct evolution of a quantum state
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
    state_in = tf.convert_to_tensor(state_in)
    cutoff = state_in.shape[1]
    
    dtype_c = state_in.dtype
    dtype_r = phi1.dtype
    
    R = tf.numpy_function(R_matrix2, [gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, state_in], dtype_c)
    state_out = R[:, :, :, 0, 0]
    
    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, state_in)"
        G = tf.numpy_function(G_matrix2, [gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, cutoff], dtype_c)
        
        dPsi_dgamma1, dPsi_dgamma1c, dPsi_dgamma2, dPsi_dgamma2c, dPsi_dphi1, dPsi_dphi2, dPsi_dtheta1, dPsi_dvarphi1, dPsi_dzeta1, dPsi_dzeta1c, dPsi_dzeta2, dPsi_dzeta2c, dPsi_dtheta, dPsi_dvarphi = tf.numpy_function(dPsi2,[gamma1, gamma2, phi1, phi2, theta1, varphi1, zeta1, zeta2, theta, varphi, state_in, G[0,0,:,:], R], (dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c, dtype_c))
        
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
        grad_Psic = tf.einsum("abcd,eab->ecd",  tf.math.conj(G), dy)
        
        return grad_gamma1c, grad_gamma2c, grad_phi1, grad_phi2, grad_theta1, grad_varphi1, grad_zeta1c, grad_zeta2c, grad_theta, grad_varphi, grad_Psic
    
    return state_out, grad

def KerrDiagonal(k, cutoff: int, dtype: tf.dtypes.DType):
    """
    Returns the diagonal of the single-mode Kerr matrix (vector)

    Arguments:
        cutoff (int): the cutoff dimension of Fock space
        dtype (tf dtype): either tf.complex64 or tf.complex128
    """
    return tf.exp(1j * tf.cast(k, dtype=dtype) * np.arange(cutoff) ** 2)
    
