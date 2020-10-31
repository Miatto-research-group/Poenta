import numpy as np
import tensorflow as tf
from poenta.circuit import Circuit
import time
import logging


from scipy.special import factorial as fac
def hex_GKP(mu, d, delta, cutoff, nmax=7):
    r"""Hexagonal GKP code state.
    The Hex GKP state is defined by
    .. math::
        |mu> = \sum_{n_1,n_2=-\infty}^\infty e^{-i(q+\sqrt{3}p)/2}
            \sqrt{4\pi/\sqrt{3}d}(dn_1+\mu) e^{iq\sqrt{4\pi/\sqrt{3}d}n_2}|0>
    where d is the dimension of a code space, \mu=0,1,...,d-1, |0> is the
    vacuum state, and the states are modulated by a Gaussian envelope in the
    case of finite energy:
    ..math:: e^{-\Delta ^2 n}|\mu>
    Args:
        d (int): the dimension of the code space.
        mu (int): mu=0,1,...,d-1.
        delta (float): width of the modulating Gaussian envelope.
        cutoff (int): the Fock basis truncation of the returned state vector.
        nmax (int): the Hex GKP state |mu> is calculated by performing the
            sum using n1,n1=-nmax,...,nmax.
    Returns:
        array: a size [cutoff] complex array state vector.
    """
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    n1sq = n1**2
    n2sq = n2**2

    sqrt3 = np.sqrt(3)

    arg1 = -1j*np.pi*n2*(d*n1+mu)/d
    arg2 = -np.pi*(d**2*n1sq+n2sq-d*n1*(n2-2*mu)-n2*mu+mu**2)/(sqrt3*d)
    arg2 *= 1-np.exp(-2*delta**2)

    amplitude = (np.exp(arg1)*np.exp(arg2)).flatten()[:, None]

    alpha = np.sqrt(np.pi/(2*sqrt3*d)) * (sqrt3*(d*n1+mu) - 1j*(d*n1-2*n2+mu))
    alpha *= np.exp(-delta**2)

    alpha = alpha.flatten()[:, None]
    n = np.arange(cutoff)[None, :]
    coherent = np.exp(-0.5*np.abs(alpha)**2)*alpha**n/np.sqrt(fac(n))

    hex_state = np.sum(amplitude*coherent, axis=0)
    return hex_state/np.linalg.norm(hex_state)
    
    
    

scheduler_status = False

logging.basicConfig(filename='GKPstate_test.log',level=logging.INFO,format='%(asctime)s %(message)s')
logging.info('Now start to traning~~~~ with loss function: 1 - <out|target>^2')
logging.info("scheduler status is: %s",scheduler_status)

state_in = np.zeros(100, dtype=np.complex128)
state_in[0] = 1
target_out = hex_GKP(1, 2, 0.3, 100, nmax=7)

def loss_2(target, output):
    return  1 - tf.abs(tf.reduce_sum(output * tf.math.conj(target)))**2

def loss_1(target, output):
    return  1 - tf.abs(tf.reduce_sum(output * tf.math.conj(target_out)))
    


_time_min = 1000000
_loss_min = 100
for num_seed in np.arange(1,700,3):
    logging.info("number of seed is: %s",num_seed)
    device = Circuit(num_layers=25, num_modes=1, num_seed = num_seed, dtype=tf.complex128)
    tuple_in_out = (state_in,target_out),
    device.set_input_output_pairs(*tuple_in_out)
    start = time.time()
    device.optimize(loss_2, steps=5000, scheduler=scheduler_status)
    end = time.time()
    runtime_now = end-start
    loss_now = device._historycallback.losses[-1]
    if _time_min > runtime_now:
        _time_min = runtime_now
    if _loss_min > loss_now:
        _loss_min = loss_now
    logging.info("the runtime is: %f" ,runtime_now)
    logging.info("the loss is: %f", loss_now)
logging.info("========================")
logging.info("===time_min is %f====",_time_min)
logging.info("===loss_min is %f====",_loss_min)

