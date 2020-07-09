import tensorflow as tf
import numpy as np
from Poenta import R_matrix, G_matrix, grad_newstate

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
    Psi_new = R[:,0]
    
    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma, phi, z, Psi)"
        G = G_matrix(gamma, phi, z, cutoff, Psi.dtype)
        dPsi_dgamma, dPsi_dgammac, dPsi_dphi, dPsi_dz, dPsi_dzc = grad_newstate(gamma, phi, z, Psi, G[0], R)
        grad_gammac = tf.reduce_sum(dy*np.conj(dPsi_dgamma) + tf.math.conj(dy)*dPsi_dgammac)
        grad_phi = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dphi)))
        grad_zc = tf.reduce_sum(dy*np.conj(dPsi_dz) + tf.math.conj(dy)*dPsi_dzc)
        grad_Psic = tf.linalg.matvec(G, dy, adjoint_a=True)
        return grad_gammac, grad_phi, grad_zc, grad_Psic
    
    return Psi_new, grad

def kerr(k, cutoff, dtype):
    diag = tf.exp(1j*tf.cast(k, dtype=dtype)*np.arange(cutoff)**2)
    return diag


def init_complex(layers:int, scale:float=0.01):
    """
    Returns the complex initialization values for a given number of layers

    Arguments:
        layers (int): number of layers
        scale (float): the std of the normal distribution from which the values are drawn

    Returns:
        (array[complex]): the vector of random complex initialization values
    """
    return np.random.normal(scale=scale, size=layers) + 1j*np.random.normal(scale=scale, size=layers)

def init_real(layers:int, scale:float=0.01):
    """
    Returns the real initialization values for a given number of layers

    Arguments:
        layers (int): number of layers
        scale (float): the std of the normal distribution from which the values are drawn

    Returns:
        (array[float]): the vector of random real initialization values
    """
    return np.random.normal(scale=scale, size=layers)

