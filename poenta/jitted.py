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

@njit#(nb.types.Tuple((nb.complex128, nb.complex128[:], nb.complex128[:,:]))(nb.complex128, nb.float64, nb.complex128))
def C_mu_Sigma(gamma: np.complex, phi:np.float, z:np.complex) -> tuple:
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

    C = np.exp(
        -0.5 * np.abs(gamma) ** 2 - 0.5 * cgamma ** 2 * exp2phidelta * tanhr
    ) / np.sqrt(coshr)
    mu = np.array(
        [
            cgamma * exp2phidelta * tanhr + gamma,
            -cgamma * eiphi / coshr,
        ]
    )
    Sigma = np.array(
        [
            [exp2phidelta * tanhr, -eiphi / coshr],
            [-eiphi / coshr, -np.exp(-1j * delta) * tanhr],
        ]
    )

    return C, mu, Sigma


@njit
def dC_dmu_dSigma(gamma: np.complex, phi:np.float, z:np.complex) -> tuple:
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
        dC_ddelta_over_r = (
            -0.5j * cgamma ** 2 * exp2phidelta * (1 - r ** 2 / 3.0 + 2 * r ** 4 / 15.0)
        ) * C
    dC_dz = eideltac * (dC_dr - 1j * dC_ddelta_over_r) / 2
    dC_dzc = eidelta * (dC_dr + 1j * dC_ddelta_over_r) / 2
    dC = np.array([dC_dgamma, dC_dgammac, dC_dphi, dC_dz, dC_dzc])

    # dmu
    dmu_dgamma = np.array([1.0, 0.0], dtype=np.complex128)
    dmu_dgammac = np.array([exp2phidelta * tanhr, -eiphi / coshr])
    dmu_dphi = np.array(
        [2j * cgamma * exp2phidelta * tanhr, -1j * eiphi / coshr]
    )
    dmu_dr = np.array(
        [
            cgamma * exp2phidelta / coshr ** 2,
            cgamma * eiphi * tanhr / coshr,
        ]
    )
    dmu_ddelta = np.array([1j * cgamma * exp2phidelta * tanhr, 0.0])
    if r > 0.01:
        dmu_ddelta_over_r = dmu_ddelta / r
    else:  # Taylor series for tanh(r)/r
        dmu_ddelta_over_r = np.array(
            [1j * cgamma * exp2phidelta * (1 - r ** 2 / 3.0 + 2 * r ** 4 / 15.0), 0.0]
        )
    dmu_dz = eideltac * (dmu_dr - 1j * dmu_ddelta_over_r) / 2
    dmu_dzc = eidelta * (dmu_dr + 1j * dmu_ddelta_over_r) / 2
    dmu = np.stack((dmu_dgamma, dmu_dgammac, dmu_dphi, dmu_dz, dmu_dzc), axis=-1)

    # dSigma
    dSigma_dgamma = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    dSigma_dgammac = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    dSigma_dphi = np.array(
        [
            [2j * exp2phidelta * tanhr, -1j * eiphi / coshr],
            [-1j * eiphi / coshr, 0.0],
        ]
    )
    dSigma_dr = np.array(
        [
            [exp2phidelta / coshr ** 2, eiphi * tanhr / coshr],
            [eiphi * tanhr / coshr, -eideltac / coshr ** 2],
        ]
    )
    dSigma_ddelta = np.array(
        [[1j * exp2phidelta * tanhr, 0.0], [0.0, 1j * eideltac * tanhr]]
    )
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

    # first row of Transformation matrix
    G0[0] = C
    for n in range(1, cutoff):
        G0[n] = mu[1] / sqrt[n] * G0[n - 1] - Sigma[1, 1] * sqrt[n - 1] / sqrt[n] * G0[n - 2]

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
def G_matrix(gamma: np.complex, phi: np.float, z: np.complex, cutoff: np.int, dtype: np.dtype = np.complex128) -> np.array:
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
def grad_newstate(gamma: np.complex, phi: np.float, z: np.complex, cutoff:int, psi: np.array, G0: np.array, R: np.array) -> list:
    """
    Computes the gradient of the new state with respect to
    gamma, gamma*, phi, z, z* but not with respect to the old state

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        psi: (complex array): old state
        G0 (complex array[D]): 1st row of the G matrix
        R (complex array[D,D]): complete R matrix

    Returns:
        list[complex array[cutoff]]: gradient of the new state with respect to
                                    gamma, gamma*, phi, z, z*
    """

    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)
    cutoff = convert_scalar(cutoff)
    # print('state in grad_newstate: ', psi)
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)
    dC, dmu, dSigma = dC_dmu_dSigma(gamma, phi, z)

    dtype = psi.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))

    dR = np.zeros((cutoff, cutoff, 5), dtype=dtype)
    dG0 = np.zeros((cutoff, 5), dtype=dtype)

    # grad of first row of Transformation matrix
    dG0[0] = dC
    for n in range(cutoff - 1):
        dG0[n + 1] = (
            dmu[1] * G0[n] + mu[1] * dG0[n] - dSigma[1, 1] * sqrt[n] * G0[n - 1] - Sigma[1, 1] * sqrt[n] * dG0[n - 1]
        ) / sqrt[n + 1]

    # first row of dR matrix
    for n in range(cutoff):
        dR[0, n] = np.dot(np.transpose(dG0[: cutoff - n]), psi)
        psi = psi[1:] * sqrt[1 : cutoff - n]

    # rest of dR matrix
    for m in range(cutoff - 1):
        for k in range(cutoff - m - 1):
            dR[m + 1, k] = (
                dmu[0] * R[m, k]
                + mu[0] * dR[m, k]
                - dSigma[0, 0] * sqrt[m] * R[m - 1, k]
                - Sigma[0, 0] * sqrt[m] * dR[m - 1, k]
                - Sigma[0, 1] * dR[m, k + 1]
                - dSigma[0, 1] * R[m, k + 1]
            ) / sqrt[m + 1]

    return list(np.transpose(dR[:, 0]))




# Extras

# @jit(nopython=True)
def approx_new_state(gamma, phi, z, old_state, order=None):
    """
    Constructs the transformed state recursively and exactly
    up to the Nth Fock amplitude, indicated by the keyword argument `order`

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (np.array(complex)): State to be transformed
        order (int): Fock space dimensionality of the exact approximation

    Returns:
        (np.array(complex)): the new state which is exact up to dimension `order`

    """
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)

    cutoff = old_state.shape[0]
    dtype = old_state.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    if order is None:
        order = cutoff

    R = np.zeros((cutoff, cutoff), dtype=dtype)
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
        for n in range(max(1, order - m)):
            R[m, n] = (
                mu[0] / sqrt[m] * R[m - 1, n]
                - Sigma[0, 0] * sqrt[m - 1] / sqrt[m] * R[m - 2, n]
                - Sigma[0, 1] / sqrt[m] * R[m - 1, n + 1]
            )

    return R[:, 0]
