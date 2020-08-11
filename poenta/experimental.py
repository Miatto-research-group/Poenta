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
from .jitted import C_mu_Sigma, dC_dmu_dSigma, convert_scalar

@njit
def R_matrix(gamma, phi, z, old_state):
    """
    Directly constructs the transformed state recursively and exactly.

    Arguments:
        gamma (complex): displacement parameter
        phi (float): phase rotation parameter
        z (complex): squeezing parameter
        old_state (complex array[batch, D]): State to be transformed

    Returns:
        R (complex array[batch,D,D]): the matrix whose 1st column is the transformed state
    """
    batch, cutoff = old_state.shape
    dtype = old_state.dtype

    z = convert_scalar(z)
    phi = convert_scalar(phi)
    gamma = convert_scalar(gamma)
    
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)

    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    R = np.zeros((batch, cutoff, cutoff), dtype=dtype)
    G0 = np.zeros(cutoff, dtype=dtype)

    # first row of Transformation matrix
    G0[0] = C
    for n in range(1, cutoff):
        G0[n] = mu[1] / sqrt[n] * G0[n - 1] - Sigma[1, 1] * sqrt[n - 1] / sqrt[n] * G0[n - 2]

    # first row of R matrix
    for n in range(cutoff):
        R[..., 0, n] = np.sum(old_state * G0[: cutoff - n], axis=-1)
        old_state = old_state[:, 1:] * sqrt[1 : cutoff - n]

    # second row of R matrix
    R[:, 1, :-1] = mu[0] * R[:, 0, :-1] - Sigma[0, 1] * R[:, 0, 1:]

    # rest of R matrix
    for m in range(2, cutoff):
        R[:, m, :-m] = (mu[0] * R[:, m - 1, :-m] - Sigma[0, 0] * sqrt[m - 1] * R[:, m - 2, :-m] - Sigma[0, 1] * R[:, m - 1, 1 : -m + 1]) / sqrt[m]

    return R



@njit
def grad_newstate(gamma: np.complex, phi: np.float, z: np.complex, psi: np.array, G0: np.array, R: np.array) -> list:
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
    batch, cutoff = psi.shape
    
    C, mu, Sigma = C_mu_Sigma(gamma, phi, z)
    dC, dmu, dSigma = dC_dmu_dSigma(gamma, phi, z)

    dtype = psi.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))

    dR = np.zeros((batch, cutoff, cutoff, 5), dtype=dtype)
    dG0 = np.zeros((cutoff, 5), dtype=dtype)

    # grad of first row of Transformation matrix
    dG0[0] = dC
    for n in range(cutoff - 1):
        dG0[n + 1] = (dmu[1] * G0[n] + mu[1] * dG0[n] - dSigma[1, 1] * sqrt[n] * G0[n - 1] - Sigma[1, 1] * sqrt[n] * dG0[n - 1]) / sqrt[n + 1]

    # first row of dR matrix
    for n in range(cutoff):
        dR[:,0, n] = np.dot(psi, dG0[: cutoff - n]) 
        psi = psi[:,1:] * sqrt[1 : cutoff - n]

    # rest of dR matrix
    for m in range(cutoff - 1):
        for k in range(cutoff - m - 1):
            dR[:,m + 1, k] = (
                np.outer(R[:, m, k], dmu[0])
                + mu[0] * dR[:, m, k]
                - np.outer(R[:, m - 1, k], dSigma[0, 0] * sqrt[m])
                - Sigma[0, 0] * sqrt[m] * dR[:, m - 1, k]
                - Sigma[0, 1] * dR[:, m, k + 1]
                - np.outer(R[:, m, k + 1], dSigma[0, 1])
            ) / sqrt[m + 1]

    return list(np.transpose(dR[:,:, 0], (2, 0, 1)))