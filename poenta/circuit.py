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
from collections import ChainMap
import rich

from .keras import QuantumCircuit, LossCallback, LearningRateScheduler, ProgressBarCallback, LossHistoryCallback


class Circuit:
    def __init__(self, num_layers: int, dtype: tf.dtypes.DType):
        self.num_layers = num_layers
        self.dtype = dtype
        self._model: QuantumCircuit

        self.set_random_seed(665)
        self._inout_pairs: tuple
        self._cumul_steps: int = 0
        self.__should_compile = True
        self.__schash = None

    def set_random_seed(self, n: int):
        tf.random.set_seed(n)
        np.random.seed(n)

    def set_input_output_pairs(self, *pairs: tuple):
        states_in, states_out = list(zip(*pairs))
        states_in = tf.convert_to_tensor(states_in, dtype=self.dtype)
        states_out = tf.convert_to_tensor(states_out, dtype=self.dtype)
        self._inout_pairs = (states_in, states_out)
        self._model = QuantumCircuit(
            num_modes=1,
            num_layers=self.num_layers,
            batch_size=states_in.shape[0],
            cutoff=states_in.shape[1],
            dtype=self.dtype,
        )

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
                opt = ChainMap(tf.optimizers.__dict__, tfa.optimizers.__dict__)[optimizer](learning_rate)
            except KeyError as e:
                guess = [
                    opt
                    for opt in ChainMap(tf.optimizers.__dict__, tfa.optimizers.__dict__)
                    if optimizer.lower() in opt.lower() or opt.lower() in optimizer.lower()
                ]
                e.args = (f"Optimizer {optimizer} not found. Did you mean one of the following: {guess}?",)
                raise e
        elif isinstance(optimizer, tf.optimizers.Optimizer):
            opt = optimizer
        else:
            raise ValueError(
                "Optimizer can be a string (e.g. 'Adam') or an instance of an optimizer (e.g. `tf.optimizers.Adam(learning_rate=0.001)`)."
            )

        if self.should_compile(optimizer, learning_rate):
            self._model.compile(optimizer=opt, loss=loss_fn, metrics=[])

        def data():
            for i in range(steps):
                yield self._inout_pairs

        ds = tf.data.Dataset.from_generator(
            data,
            output_types=(self._model.complextype, self._model.complextype),
            output_shapes=(self._inout_pairs[0].shape, self._inout_pairs[1].shape),
        )

        history = LossHistoryCallback()
        self._model.fit(
            x=ds,
            batch_size=len(self._inout_pairs),
            steps_per_epoch=steps,
            verbose=0,
            callbacks=[LossCallback(), ProgressBarCallback(steps, self._cumul_steps), history, LearningRateScheduler(learning_rate)],
            max_queue_size=40,
            workers=1,
            use_multiprocessing=False,
        )

        self._cumul_steps += steps
        
        return history

    def show_evolution(self, state_in:tf.Tensor, figsize:tuple = (16,6), cutoff:int = 30, logy:bool = False):
        functor = tf.keras.backend.function([self._model.input], [layer.output for layer in self._model.layers])   # evaluation function
        layer_outs = functor(state_in)

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['figure.figsize'] = figsize

    
        fig, ax = plt.subplots(int(np.ceil(self.num_layers / 5)), 5)

        for k,o in enumerate(layer_outs):
            if logy:
                ax[k//5,k%5].set_yscale('log')
            ax[k//5,k%5].set_ylim([1e-6, 1.1])
            ax[k//5,k%5].bar(range(min(cutoff,self._model.cutoff)),(abs(o[0])**2)[:min(cutoff, self._model.cutoff)])

    def __repr__(self):
        circuit._model.summary(line_length=80, print_fn = rich.print)
        return ''
