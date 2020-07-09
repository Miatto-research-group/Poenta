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
