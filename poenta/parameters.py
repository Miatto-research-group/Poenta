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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from .nputils import init_complex, init_real


class Parameters:
    """
    Utility class to hold and manage the parameters of a circuit (trainable or constant).
    The `Parameters` class allows to store, visualize and share optimization results.
    As an example, the `Circuit` class contains an instance of `Parameters`.
    """

    def __init__(self, num_layers: int, dtype: tf.dtypes.DType = tf.complex64):
        """
        Arguments:
            num_layers (int): number of layers in the circuit
            dtype (tensorflow Dtype): type of the complex parameters of the circuit. The real parameters are automatically set accordingly.
        """
        if dtype == tf.complex128:
            self.complextype = tf.complex128
            self.realtype = tf.float64
        elif dtype == tf.complex64:
            self.complextype = tf.complex64
            self.realtype = tf.float32
        else:
            raise ValueError(f"dtype can be only tf.complex128 or tf.complex64, not {dtype}")

        self.gamma = tf.Variable(init_complex(num_layers, 0.01), dtype=self.complextype, name=f"gamma")
        self.phi = tf.Variable(init_real(num_layers, 0.01), dtype=self.realtype, name=f"phi")
        self.zeta = tf.Variable(init_complex(num_layers, 0.01), dtype=self.complextype, name=f"zeta")
        self.kappa = tf.Variable(init_real(num_layers, 0.01), dtype=self.realtype, name=f"kappa")

        self._history: dict = {
            "gamma": [self.gamma.numpy()],
            "phi": [self.phi.numpy()],
            "zeta": [self.zeta.numpy()],
            "kappa": [self.kappa.numpy()],
        }

    @property
    def trainable(self):
        return [var for var in (self.gamma, self.phi, self.zeta, self.kappa) if isinstance(var, tf.Variable)]

    @property
    def constant(self):
        return [var for var in (self.gamma, self.phi, self.zeta, self.kappa) if isinstance(var, tf.Tensor)]

    @property
    def all(self):
        return (self.gamma, self.phi, self.zeta, self.kappa)

    @property
    def L1_norm(self):
        tensor = tf.stack([tf.cast(var, self.complextype) for var in self.trainable])
        return tf.abs(tf.linalg.norm(tensor, ord=1))

    def plot(self, name: str, layer: int = None):
        """
        Plots the history of the parameters during the learning phase.
        Complex parameters are plotted as paths in the complex plane,
        while real parameters are plotted as graphs with respect to the step number.

        Arguments:
            name (str): the name of the parameter among 'gamma', 'phi', 'zeta' and 'kappa'
            layer (int or None): if none plot all of the layers, otherwise plot a specific layer
        """
        values = np.array(self._history[name])
        if layer is not None:
            values = values[:, layer]

        if name.lower() in {"gamma", "zeta"}:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_aspect("equal")

            xlim = 1.1 * np.max(np.abs(np.real(values)))
            ylim = 1.1 * np.max(np.abs(np.imag(values)))

            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel("Real")
            ax.set_ylabel("Imag")

            for v in np.transpose(values):
                plt.plot(np.real(v), np.imag(v))
                plt.grid()

        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.set_xlabel("step")
            ax.set_ylabel("value")

            for v in np.transpose(values):
                plt.plot(v)
            plt.grid()

        ax.set_title(name + ", all layers" if layer is None else f", layer {layer}")
        return ax

    def save(self):
        self._history["gamma"].append(self.gamma.numpy())
        self._history["phi"].append(self.phi.numpy())
        self._history["zeta"].append(self.zeta.numpy())
        self._history["kappa"].append(self.kappa.numpy())

    def __repr__(self):
        table = PrettyTable()
        table.add_column("Layer", list(range(1, 1 + self.gamma.shape[0])))
        table.add_column("gamma", [f"{val:.3f}" for val in self.gamma.numpy()])
        table.add_column("phi", [f"{val:.3f}" for val in self.phi.numpy()])
        table.add_column("zeta", [f"{val:.3f}" for val in self.zeta.numpy()])
        table.add_column("kappa", [f"{val:.3f}" for val in self.kappa.numpy()])
        return str(table)
