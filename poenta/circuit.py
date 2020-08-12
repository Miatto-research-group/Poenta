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
from typing import Callable, Union, Iterable
from dataclasses import dataclass, field
from collections import ChainMap
from prettytable import PrettyTable

from .keras import QuantumCircuit, ProgressBarCallback, LossHistoryCallback


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
    def __init__(self, num_layers: int, dtype: tf.dtypes.DType):
        self.num_layers = num_layers
        self.dtype = dtype
        self._circuit = QuantumCircuit(num_modes=1, num_layers=num_layers, dtype=dtype)
        self._random_seed = 665
        self._inout_pairs: tuple
        self.__should_compile = True
        self.__schash = None

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, n: int):
        self._random_seed = n
        tf.random.set_seed(n)
        np.random.seed(n)

    def set_input_output_pairs(self, *pairs: tuple):
        states_in, states_out = list(zip(*pairs))
        self._inout_pairs = (
            tf.convert_to_tensor(states_in, dtype=self.dtype),
            tf.convert_to_tensor(states_out, dtype=self.dtype),
        )
        self._circuit._batch_size = len(pairs)

    def should_compile(self, optimizer, learning_rate):
        _hash = hash((hash(optimizer), hash(learning_rate)))
        self.__should_compile = self.__schash != _hash
        self.__schash = _hash
        return self.__should_compile

    def optimize(
        self,
        loss_fn: Callable,
        steps: int,
        optimizer: Union[str, tf.optimizers.Optimizer] = "Adam",
        learning_rate: float = 0.001,
    ) -> LossHistoryCallback:
        if isinstance(optimizer, str):
            try:
                opt = ChainMap(tf.optimizers.__dict__, tfa.optimizers.__dict__)[optimizer.capitalize()](learning_rate)
            except KeyError:
                raise ValueError("Optimizer {optimizer} not found.")
        elif isinstance(optimizer, tf.optimizers.Optimizer):
            opt = optimizer
        else:
            raise ValueError(
                "Optimizer can be a string (e.g. 'Adam') or an instance of an optimizer (e.g. `tf.optimizers.Adam(learning_rate=0.001)`)."
            )

        if self.should_compile(optimizer, learning_rate):
            self._circuit.compile(optimizer=opt, loss=loss_fn, metrics=[])

        def data():
            for i in range(steps):
                yield self._inout_pairs

        ds = tf.data.Dataset.from_generator(
            data,
            output_types=(self._circuit.complextype, self._circuit.complextype),
            output_shapes=(self._inout_pairs[0].shape, self._inout_pairs[1].shape),
        )

        history = LossHistoryCallback()
        self._circuit.fit(
            x=ds,
            batch_size=len(self._inout_pairs),
            steps_per_epoch=steps,
            verbose=0,
            callbacks=[ProgressBarCallback(steps), history],
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        return history

    def export_weights(self, filename: str):
        pass

    # def __repr__(self):
    #     table = PrettyTable()
    #     table.add_column("Layers", [self.num_layers])
    #     table.add_column("Cutoff", [self.cutoff])
    #     table.add_column("Optimizer", [self.config.optimizer])
    #     trainable_pars = np.sum([p.shape for p in self.parameters.trainable])
    #     tot_pars = np.sum([p.shape for p in self.parameters.all])
    #     table.add_column("Params (trainable/tot)", [f"{trainable_pars}/{tot_pars}"])
    #     return str(table)
