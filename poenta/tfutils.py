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

from .jitted import G_matrix, R_matrix, grad_newstate

from .jitted import G_matrix, grad_newstate
from .experimental import R_matrix


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
def GaussianTransformation(gamma: tf.Variable, phi: tf.Variable, z: tf.Variable, state_in: tf.Tensor, cutoff: int) -> tf.Tensor:
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
    state_in = tf.convert_to_tensor(tf.squeeze(state_in))

    dtype_c = state_in.dtype
    dtype_r = phi.dtype
    # print('about to call R')
    R = tf.numpy_function(R_matrix, [gamma, phi, z, cutoff, state_in], dtype_c)
    state_out = R[:, 0]
    # print('called R')
    # print(f"post: {gamma}\n {phi}\n {z}\n {state_in}\n {state_out}")

    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma, phi, z, Psi)"
        G = tf.numpy_function(G_matrix, [gamma, phi, z, cutoff], dtype_c)
        dPsi_dgamma, dPsi_dgammac, dPsi_dphi, dPsi_dz, dPsi_dzc = tf.numpy_function(
            grad_newstate, [gamma, phi, z, cutoff, state_in, G[0], R], (dtype_c, dtype_c, dtype_c, dtype_c, dtype_c)
        )
        grad_gammac = tf.reduce_sum(dy * tf.math.conj(dPsi_dgamma) + tf.math.conj(dy) * dPsi_dgammac)
        grad_phi = 2 * tf.math.real(tf.reduce_sum(dy * tf.math.conj(dPsi_dphi)))
        grad_zc = tf.reduce_sum(dy * tf.math.conj(dPsi_dz) + tf.math.conj(dy) * dPsi_dzc)
        grad_Psic = tf.linalg.matvec(G, dy, adjoint_a=True)  # NOTE: can we compute directly the product between G and dy?
        return grad_gammac, grad_phi, grad_zc, grad_Psic, None

    return state_out, grad


def KerrDiagonal(k, cutoff: int, dtype: tf.dtypes.DType):
    """
    Returns the diagonal of the single-mode Kerr matrix

    Arguments:
        cutoff (int): the cutoff dimension of Fock space
        dtype (tf dtype): either tf.complex64 or tf.complex128
    """
    return tf.exp(1j * tf.cast(k, dtype=dtype) * np.arange(cutoff) ** 2)
