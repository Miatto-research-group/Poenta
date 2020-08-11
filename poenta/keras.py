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
from .tfutils import complex_initializer, real_initializer, GaussianTransformation, KerrDiagonal


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_modes: int, cutoff: int, dtype: tf.dtypes.DType):
        super().__init__()
        self.num_modes = num_modes
        self.cutoff = cutoff
        if dtype == tf.complex128:
            self.complextype = tf.complex128
            self.realtype = tf.float64
        elif dtype == tf.complex64:
            self.complextype = tf.complex64
            self.realtype = tf.float32
        else:
            raise ValueError("dtype can be only tf.complex64 or tf.complex128")

    def build(self, input_shape):  # TODO: upgrade for 2 modes
        self.gamma = self.add_weight("gamma", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype))
        self.phi = self.add_weight("phi", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype))
        self.zeta = self.add_weight("zeta", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype))
        self.kappa = self.add_weight("kappa", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype))
        super().build(input_shape)  # is this necessary?

    def call(self, input):
        gaussian_output = GaussianTransformation(self.gamma, self.phi, self.zeta, input, self.cutoff)
        return KerrDiagonal(self.kappa, self.cutoff, dtype=self.complextype) * gaussian_output



class QuantumDevice(tf.keras.Model):  # or Sequential?
    def __init__(self, num_modes, num_layers, cutoff, dtype):
        super().__init__(name="")
        self._layers = [QuantumLayer(num_modes, cutoff, dtype) for _ in range(num_layers)]

    def call(self, input_tensor):
        for layer in self._layers:
            input_tensor = layer(input_tensor)
        return input_tensor
      
    @property
    def num_layers(self):
        return len(self._layers)

    @property
    def cutoff(self):
        return self._layers[0].cutoff

