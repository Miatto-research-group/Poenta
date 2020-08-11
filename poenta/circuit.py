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
import tensorflow_addons as tfa
import numpy as np
from typing import Callable
from dataclasses import dataclass, field
from collections import ChainMap
from prettytable import PrettyTable

from .keras import QuantumDevice, ProgressBarCallback, LossHistoryCallback


@dataclass
class OptimizationConfig:
    state_in: np.array
    objective: np.array
    loss_fn: Callable
    optimizer: str = "Adam"
    random_seed: int = 665
    LR: float = 0.001
    LR_schedule: dict = field(default_factory=dict)  # optional


class Circuit:
    def __init__(self, num_layers: int, cutoff: int, dtype: tf.dtypes.DType, config: OptimizationConfig):

        self.num_layers = num_layers
        self.cutoff = cutoff
        self.dtype = dtype
        self._circuit = QuantumDevice(num_modes=1, num_layers=num_layers, cutoff=cutoff, dtype=dtype)
        self.set_optimization_config(config)

    def set_optimization_config(self, config: OptimizationConfig):
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
        self.config = config
        self.state_in = tf.cast(tf.constant(config.state_in), dtype=self.dtype)
        self.objective = tf.cast(tf.constant(config.objective), dtype=self.dtype)
        self.optimizer = ChainMap(tf.optimizers.__dict__, tfa.optimizers.__dict__)[config.optimizer](config.LR)
        self.loss_fn = config.loss_fn
        self._circuit.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[])

    def reset(self):
        self.circuit = QuantumDevice(num_modes=1, num_layers=num_layers, cutoff=cutoff, dtype=dtype)
        self.set_optimization_config(self.config)

    def optimize(self, steps: int, epochs: int = 1) -> list:
        def data():
            for i in range(steps):
                yield (self.state_in, self.objective)

        ds = tf.data.Dataset.from_generator(
            data,
            output_types=(self._circuit.complextype, self._circuit.complextype),
            output_shapes=(tf.TensorShape([self.cutoff]), tf.TensorShape([self.cutoff])),
        )
        history = LossHistoryCallback()
        self._circuit.fit(
            x=ds.repeat(epochs), batch_size=1, steps_per_epoch=steps, verbose = 0, callbacks=[ProgressBarCallback(steps, epochs), history], epochs=epochs, workers=1, use_multiprocessing=False
        )
        return history

    
    # def __repr__(self):
    #     table = PrettyTable()
    #     table.add_column("Layers", [self.num_layers])
    #     table.add_column("Cutoff", [self.cutoff])
    #     table.add_column("Optimizer", [self.config.optimizer])
    #     trainable_pars = np.sum([p.shape for p in self.parameters.trainable])
    #     tot_pars = np.sum([p.shape for p in self.parameters.all])
    #     table.add_column("Params (trainable/tot)", [f"{trainable_pars}/{tot_pars}"])
    #     return str(table)
