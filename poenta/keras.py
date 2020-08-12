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
from .tfutils import real_complex_types, complex_initializer, real_initializer, GaussianTransformation, KerrDiagonal
from .parameters import Parameters
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_modes: int, realtype: tf.dtypes.DType, complextype: tf.dtypes.DType):
        super().__init__()
        self.num_modes = num_modes
        self.realtype = realtype
        self.complextype = complextype

    def build(self, input_shape):  # TODO: upgrade for 2 modes
        self.gamma = self.add_weight("gamma", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype))
        self.phi = self.add_weight("phi", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype))
        self.zeta = self.add_weight("zeta", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype))
        self.kappa = self.add_weight("kappa", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype))
        super().build(input_shape)  # is this necessary?

    def call(self, input):
        gaussian_output = GaussianTransformation(self.gamma, self.phi, self.zeta, input)
        output = KerrDiagonal(self.kappa, input.shape[1], dtype=self.complextype)[None,:] * gaussian_output
        output.set_shape(input.get_shape())
        return output


class QuantumCircuit(tf.keras.Model):  # or Sequential?
    def __init__(self, num_modes, num_layers, dtype):
        super().__init__(name="")
        self.realtype, self.complextype = real_complex_types(dtype)
        self._layers = [QuantumLayer(num_modes, self.realtype, self.complextype) for _ in range(num_layers)]
        self._loss = 1.0
        self._batch_size = None
    
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size
        for layer in self._layers:
            layer.batch_size = batch_size

    def call(self, input_tensor):
        for layer in self._layers:
            input_tensor = layer(input_tensor)
        return input_tensor

    @property
    def num_layers(self):
        return len(self._layers)


class ProgressBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, steps: int, epochs: int):
        super().__init__()
        self.steps = steps
        self.epochs = epochs
        self.task = None
        if epochs > 1:
            first_field=TextColumn("[progress.description]Epoch {task.fields[epoch]} | Iteration {task.fields[iteration]}/{task.total}")
        else:
            first_field=TextColumn("Iteration {task.fields[iteration]}/{task.total}")
        self.bar = Progress(
        first_field,
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("Loss = {task.fields[loss]:.5f} | Time remaining: "),
        TimeRemainingColumn(),
    )
        self.prev_avg_loss = None
        self.current_avg_loss = None
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        self.current_avg_loss = self.model._loss
        self.task = self.bar.add_task(description="Optimizing...", total=self.steps*self.epochs, iteration=0, loss=self.current_avg_loss, epoch=0)

    def on_train_end(self, logs=None):
        self.model._loss = self.current_loss(self.steps*self.epochs)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch+1
        self.bar.update(self.task, epoch=epoch+1)

    def on_train_batch_begin(self, batch, logs=None):
        self.prev_avg_loss = self.current_avg_loss

    def on_train_batch_end(self, batch, logs=None):
        self.current_avg_loss = logs['loss']
        self.bar.update(self.task, advance=1, refresh=True, iteration=(self.current_epoch-1)*self.steps + batch + 1, loss=self.current_loss(batch))

    def current_loss(self, batch):
        return (batch+1)*self.current_avg_loss - batch*self.prev_avg_loss
        


class LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.prev_avg_loss = None
        self.current_avg_loss = None
        
    def on_train_begin(self, logs=None):
        self.current_avg_loss = self.model._loss
        self.losses = []
        self.val_losses = []

    def on_train_batch_begin(self, batch, logs=None):
        self.prev_avg_loss = self.current_avg_loss

    def on_train_batch_end(self, batch, logs=None):
        self.current_avg_loss = logs['loss']
        self.losses.append(self.current_loss(batch))

    def current_loss(self, batch):
        return (batch+1)*self.current_avg_loss - batch*self.prev_avg_loss