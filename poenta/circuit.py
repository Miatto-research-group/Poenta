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
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from dataclasses import dataclass, field
from collections import ChainMap
from prettytable import PrettyTable

from .nputils import init_complex, init_real
from .tfutils import GaussianTransformation, kerr
from .parameters import Parameters


@dataclass
class OptimizationConfig:
    state_in: np.array
    objective: np.array
    optimizer: str = "Adam"
    random_seed: int = 666
    LR: float = 0.001
    LR_schedule: dict = field(default_factory=dict)  # optional


class Circuit:
    def __init__(self, num_layers: int, dtype: tf.dtypes.DType, config: OptimizationConfig):
        self.dtype = dtype
        self.set_optimization_config(config)

        self.num_layers = num_layers
        self._state_out = None
        self._fidelity = None
        self.cutoff = self.state_in.shape[0]
        self.parameters = Parameters(num_layers=num_layers, dtype=dtype)

    def set_optimization_config(self, config: OptimizationConfig):
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
        self.config = config
        self.state_in = tf.cast(tf.constant(config.state_in), dtype=self.dtype)
        self.objective = tf.cast(tf.constant(config.objective), dtype=self.dtype)
        self.optimizer = ChainMap(tf.optimizers.__dict__, tfa.optimizers.__dict__)[config.optimizer](config.LR)

    def _layer_out(self, gamma: tf.Tensor, phi: tf.Tensor, zeta: tf.Tensor, k: tf.Tensor, layer_in: tf.Tensor) -> tf.Tensor:
        layer_out = GaussianTransformation(gamma, phi, zeta, layer_in)
        return kerr(k, self.cutoff, self.dtype) * layer_out

    @property  # lazy property
    def state_out(self) -> tf.Tensor:
        if self._state_out is None:
            state = self.state_in
            for i in range(self.num_layers):
                state = self._layer_out(
                    self.parameters.gamma[i],
                    self.parameters.phi[i],
                    self.parameters.zeta[i],
                    self.parameters.kappa[i],
                    state,
                )
            self._state_out = state
        return self._state_out

    @property  # lazy property
    def fidelity(self):
        if self._fidelity is None:
            self._fidelity = tf.abs(tf.reduce_sum(self.state_out * tf.math.conj(self.objective))) ** 2
        return self._fidelity

    def loss(self):
        return 1.0 - self.fidelity

    def minimize_step(self) -> float:
        self._state_out = None  # reset lazy output state
        self._fidelity = None # reset lazy fidelity
        self.parameters.save()  # save current values before updating
        self.optimizer.minimize(self.loss, self.parameters.trainable)
        return self.loss()

    def minimize(self, steps) -> list:
        loss_list = [] 

        with Progress(
            TextColumn("[progress.description] Iteration {task.fields[iteration]}/{task.total}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Loss = {task.fields[loss]:.5f} | Time remaining: "),
            TimeRemainingColumn()) as bar:
            task = bar.add_task(description="Optimizing...", total=steps, iteration=0, loss = self.loss())
            for i in range(steps):
                try:
                    loss_list.append(self.minimize_step())

                    # LR scheduling
                    for threshold, new_LR in self.config.LR_schedule.items():
                        if loss_list[-1] < threshold:
                            self.optimizer.lr = new_LR

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"other exception: {e}")
                    raise e
                bar.update(task, advance=1, refresh=True, iteration = i+1, loss = self.loss())
        return loss_list

    def __repr__(self):
        table = PrettyTable()
        table.add_column("Layers", [self.num_layers])
        table.add_column("Cutoff", [self.cutoff])
        table.add_column("Optimizer", [self.config.optimizer])
        trainable_pars = np.sum([p.shape for p in self.parameters.all])
        tot_pars = np.sum([p.shape for p in self.parameters.trainable])
        table.add_column("Params (trainable/tot)", [f"{trainable_pars}/{tot_pars}"])
        return str(table)
