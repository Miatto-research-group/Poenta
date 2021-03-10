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
from scipy.special import factorial as fac


def init_complex(layers: int, scale: float = 0.01):
    """
    Returns the complex initialization values for a given number of layers

    Arguments:
        layers (int): number of layers
        scale (float): the std of the normal distribution from which the values are drawn

    Returns:
        (array[complex]): the vector of random complex initialization values
    """
    return np.random.normal(scale=scale, size=layers) + 1j * np.random.normal(scale=scale, size=layers)


def init_real(layers: int, scale: float = 0.01):
    """
    Returns the real initialization values for a given number of layers

    Arguments:
        layers (int): number of layers
        scale (float): the std of the normal distribution from which the values are drawn

    Returns:
        (array[float]): the vector of random real initialization values
    """
    return np.random.normal(scale=scale, size=layers)





##############States###############

def vaccum(mode,cutoff):
    if mode == 1:
        state =np.zeros([cutoff])
        state[0] = 1
    elif mode == 2:
        state = np.zeros((cutoff,cutoff), dtype=np.complex128)
        state[0,0] = 1
    return state

def single_photon(mode,cutoff):
    if mode == 1:
        state = np.zeros([cutoff])
        state[1] = 1
    return state


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

def NOON(N, cutoff):
    state = np.zeros((cutoff, cutoff))
    state[0, N] = 1/np.sqrt(2)
    state[N, 0] = 1/np.sqrt(2)
    return state
