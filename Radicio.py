import tensorflow as tf
import numpy as np
from Poenta import R_matrix, G_matrix, grad_newstate

@tf.custom_gradient
def GaussianTransformation(gamma, phi, z, Psi):
    """
    Direct evolution of a quantum state
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
        G = G_matrix(gamma, phi, z, cutoff)
        dPsi_dgamma, dPsi_dgammac, dPsi_dphi, dPsi_dz, dPsi_dzc = grad_newstate(gamma, phi, z, Psi, G[0], R)
        grad_gammac = tf.reduce_sum(dy*np.conj(dPsi_dgamma) + tf.math.conj(dy)*dPsi_dgammac)
        grad_phi = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dphi)))
        grad_zc = tf.reduce_sum(dy*np.conj(dPsi_dz) + tf.math.conj(dy)*dPsi_dzc)
        grad_Psic = tf.linalg.matvec(G, dy, adjoint_a=True)
        return grad_gammac, grad_phi, grad_zc, grad_Psic
    
    return Psi_new, grad

def kerr(k, cutoff):
    diag = tf.exp(1j*tf.cast(k, dtype=tf.complex128)*np.arange(cutoff)**2)
    return diag


def init_complex(layers, scale=0.01):
    return np.random.normal(scale=scale, size=layers) + 1j*np.random.normal(scale=scale, size=layers)

def init_real(layers, scale=0.01):
    return np.random.normal(scale=scale, size=layers)

