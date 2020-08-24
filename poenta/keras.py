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
from .tfutils import real_complex_types, complex_initializer, real_initializer, GaussianTransformation, KerrDiagonal, GaussianTransformation2mode, KerrDiagonalT
from .parameters import Parameters
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from numpy import pi, cos, tanh


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_modes: int, cutoff: int, realtype: tf.dtypes.DType, complextype: tf.dtypes.DType):
        super().__init__()
        self.num_modes = num_modes
        self.cutoff = cutoff
        self.realtype = realtype
        self.complextype = complextype

    def build(self, input_shape):  # TODO: upgrade for 2 modes
        if self.num_modes == 1:
            self.gamma = self.add_weight(
                "gamma", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype)
            )
            self.phi = self.add_weight(
                "phi", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.zeta = self.add_weight(
                "zeta", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype)
            )
            self.kappa = self.add_weight(
                "kappa", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
        elif self.num_modes == 2:
            self.gamma1 = self.add_weight(
                "gamma1", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype)
            )
            self.gamma2 = self.add_weight(
                "gamma2", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype)
            )
            self.phi1 = self.add_weight(
                "phi1", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.phi2 = self.add_weight(
                "phi2", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.theta1 = self.add_weight(
                "theta1", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.varphi1 = self.add_weight(
                "varphi1", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.zeta1 = self.add_weight(
                "zeta1", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype)
            )
            self.zeta2 = self.add_weight(
                "zeta2", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype)
            )
            self.theta = self.add_weight(
                "theta", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.varphi = self.add_weight(
                "varphi", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.kappa1 = self.add_weight(
                "kappa1", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
            self.kappa2 = self.add_weight(
                "kappa2", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype)
            )
        super().build(input_shape)  # is this necessary?

    def call(self, input):
        if self.num_modes == 1:
            gaussian_output = GaussianTransformation(self.gamma, self.phi, self.zeta, input)
            output = KerrDiagonal(self.kappa, self.cutoff, dtype=self.complextype)[None, :] * gaussian_output
        elif self.num_modes == 2:
            gaussian_output = GaussianTransformation2mode(self.gamma1, self.gamma2, self.phi1, self.phi2, self.theta1, self.varphi1, self.zeta1, self.zeta2, self.theta, self.varphi, input)
            output = KerrDiagonalT(self.kappa1, self.cutoff, dtype=self.complextype) * gaussian_output * KerrDiagonal(self.kappa2, self.cutoff, dtype=self.complextype)
        output.set_shape(input.get_shape())
        return output


class QuantumCircuit(tf.keras.Sequential):
    def __init__(self, num_modes: int, num_layers: int, batch_size: int, cutoff: int, dtype: tf.DType):
        self.realtype, self.complextype = real_complex_types(dtype)
        self._loss = 1.0
        self._batch_size = batch_size
        self._tot_batches = 0
        self.cutoff = cutoff
        if num_modes == 1:
            super().__init__(
                [tf.keras.Input(shape=[cutoff], batch_size=batch_size, dtype=dtype)]
                + [QuantumLayer(num_modes, cutoff, self.realtype, self.complextype) for _ in range(num_layers)]
            )
        elif num_modes == 2:
            super().__init__(
                [tf.keras.Input(shape=[cutoff,cutoff], batch_size=batch_size, dtype=dtype)]
                + [QuantumLayer(num_modes, cutoff, self.realtype, self.complextype) for _ in range(num_layers)]
            )


class LossCallback(tf.keras.callbacks.Callback):
    """
    Callback that embeds the loss value as a model attribute 
    at the end of each training batch.
    """
    def __init__(self):
        super().__init__()
        self.prev_avg_loss = None
        self.current_avg_loss = None

    def on_train_begin(self, logs=None):
        self.current_avg_loss = self.model._loss

    def on_train_batch_begin(self, batch, logs=None):
        self.prev_avg_loss = self.current_avg_loss

    def on_train_batch_end(self, batch, logs=None):
        self.current_avg_loss = logs["loss"]
        self.model._loss = self.current_loss(batch)

    def current_loss(self, batch):
        return (batch + 1) * self.current_avg_loss - batch * self.prev_avg_loss



class ProgressBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, steps: int):
        super().__init__()
        self.steps = steps
        self.task = None
        self.bar = Progress(
            TextColumn("Iteration {task.fields[iteration]}/{task.fields[cumul]}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Loss = {task.fields[loss]:.5f} | ⏳ "),
            TimeRemainingColumn(),
            TextColumn("lr = {task.fields[lr]:.6f}"),
        )

    def on_train_begin(self, logs=None):
        self.task = self.task = self.bar.add_task(
            description="Optimizing...", total=self.steps, iteration=self.model._tot_batches, loss=self.model._loss, cumul=self.steps+self.model._tot_batches, lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        )

    def on_train_batch_end(self, batch, logs=None):
        self.model._tot_batches += 1
        self.bar.update(self.task, advance=1, refresh=True, iteration=self.model._tot_batches, loss=self.model._loss, lr=float(tf.keras.backend.get_value(self.model.optimizer.lr)))



class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr:float, min_lr:float = 0.00001):
        super().__init__()
        self.initial_lr = initial_lr
        self.epsilon = initial_lr*(1-tanh(10.0))
        
    def on_train_batch_begin(self, batch, logs=None):
        new_lr = self.initial_lr*tanh(10.0*self.model._loss) + self.epsilon
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)


class LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses= []

    def on_train_batch_end(self, batch, logs=None):
        self.losses.append(self.model._loss)


