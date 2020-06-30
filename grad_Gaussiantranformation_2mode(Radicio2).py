import tensorflow as tf
import numpy as np
from Poenta import R_matrix, G_matrix, grad_newstate

@tf.custom_gradient
def GaussianTransformation2mode(gamma, phi, theta1, varphi1, zeta, theta, varphi, Psi):
    """
    Direct evolution of a quantum state
    """
    gamma = gamma.numpy()
    phi = phi.numpy()
    z = z.numpy()
    Psi = Psi.numpy()
    cutoff = Psi.shape[0]
    
    R = R_matrix(gamma, phi, theta1, varphi1, zeta, theta, varphi, Psi)
    Psi_new = R[:,:,0,0]
    
    def grad(dy):
        "Vector-Jacobian products for all the arguments (gamma, phi, theta1, varphi1, zeta, theta, varphi, Psi)"
        G = G_matrix(gamma, phi, theta1, varphi1, zeta, theta, varphi, cutoff)
        dPsi_dgamma1, dPsi_dgamma1c, dPsi_dgamma2, dPsi_dgamma2c, dPsi_dphi1, dPsi_dphi2, dPsi_dtheta1, dPsi_dvarphi1, dPsi_dzeta1, dPsi_dzeta1c, dPsi_dzeta2, dPsi_dzeta2c, dPsi_dtheta, dPsi_dvarphi= grad_newstate(gamma, phi, theta1, psi1, zeta, theta, psi, Psi, G[0,0,:,:], R)
        
        grad_gamma1c = tf.reduce_sum(dy*np.conj(dPsi_dgamma1) + tf.math.conj(dy)*dPsi_dgamma1c)
        grad_gamma2c = tf.reduce_sum(dy*np.conj(dPsi_dgamma2) + tf.math.conj(dy)*dPsi_dgamma2c)
        grad_zeta1c = tf.reduce_sum(dy*np.conj(dPsi_dzeta1) + tf.math.conj(dy)*dPsi_dzeta1c)
        grad_zeta2c = tf.reduce_sum(dy*np.conj(dPsi_dzeta2) + tf.math.conj(dy)*dPsi_dzeta2c)
        grad_phi1 = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dphi1)))
        grad_phi2 = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dphi2)))
        grad_theta1 = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dtheta1)))
        grad_varphi1 = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dvarphi1)))
        grad_theta = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dtheta)))
        grad_varphi = 2*tf.math.real(tf.reduce_sum(dy*np.conj(dPsi_dvarphi)))
        grad_Psic = tf.linalg.matvec(G, dy, adjoint_a=True)
        return grad_gamma1c, grad_gamma2c, grad_phi1, grad_phi2, grad_theta1, grad_varphi1, grad_zeta1c, grad_zeta2c, grad_theta, grad_varphi, grad_Psic
    
    return Psi_new, grad

def kerr(k, cutoff):
    diag = tf.exp(1j*tf.cast(k, dtype=tf.complex128)*np.arange(cutoff)**2)
    return diag


def init_complex(layers, scale=0.01):
    return np.random.normal(scale=scale, size=layers) + 1j*np.random.normal(scale=scale, size=layers)

def init_real(layers, scale=0.01):
    return np.random.normal(scale=scale, size=layers)


