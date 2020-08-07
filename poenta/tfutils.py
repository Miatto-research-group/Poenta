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
from .jitted import G_matrix, grad_newstate, R_matrix


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


@tf.custom_gradient
def GaussianTransformation(gamma, phi, z, Psi):
    """
    Evolution of a single-mode quantum state through a Gaussian transformation parametrized
    by the three parameters gamma, phi and z.
    This is a differentiable function that can be safely used in a TensorFlow computation.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation
        z (complex): squeezing parameter
        Psi (complex array[D]): input state of dimension D
    
    Returns:
        Psi_new (complex array[D]): output state of dimension D
    """
    gamma = gamma.numpy()
    phi = phi.numpy()
    z = z.numpy()
    Psi = Psi.numpy()
    cutoff = Psi.shape[0]

    R = R_matrix(gamma, phi, z, Psi)
    Psi_new = R[:, 0]

    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma, phi, z, Psi)"
        G = G_matrix(gamma, phi, z, cutoff, Psi.dtype)
        dPsi_dgamma, dPsi_dgammac, dPsi_dphi, dPsi_dz, dPsi_dzc = grad_newstate(gamma, phi, z, Psi, G[0], R)
        grad_gammac = tf.reduce_sum(dy * np.conj(dPsi_dgamma) + tf.math.conj(dy) * dPsi_dgammac)
        grad_phi = 2 * tf.math.real(tf.reduce_sum(dy * np.conj(dPsi_dphi)))
        grad_zc = tf.reduce_sum(dy * np.conj(dPsi_dz) + tf.math.conj(dy) * dPsi_dzc)
        grad_Psic = tf.linalg.matvec(
            G, dy, adjoint_a=True
        )  # NOTE: can we compute directly the product between G and dy?
        return grad_gammac, grad_phi, grad_zc, grad_Psic

    return Psi_new, grad


def kerr(k, cutoff: int, dtype: tf.dtypes.DType):
    """
    Returns the diagonal of the single-mode Kerr matrix

    Arguments:
        cutoff (int): the cutoff dimension of Fock space
        dtype (tf dtype): either tf.complex64 or tf.complex128
    """
    diag = tf.exp(1j * tf.cast(k, dtype=dtype) * np.arange(cutoff) ** 2)
    return diag
