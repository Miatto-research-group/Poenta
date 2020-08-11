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
from .jitted import C_mu_Sigma, convert_scalar



@njit
def R_matrix(gamma, phi, z, cutoff, old_state):
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

    # second row of R matrix
    R[1, :-1] = mu[0] * R[0, :-1] - Sigma[0, 1] * R[0, 1:]

    # rest of R matrix
    for m in range(2, cutoff):
        R[m, :-m] = (mu[0] * R[m - 1, :-m]
                    - Sigma[0, 0] * sqrt[m - 1] * R[m - 2, :-m]
                    - Sigma[0, 1] * R[m - 1, 1:-m+1]) / sqrt[m]

    return R#[:, 0]