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
from tqdm.notebook import tqdm
from dataclasses import dataclass, field
from typing import Union

from nputils import init_complex, init_real
from tfutils import GaussianTransformation, kerr
from parameters import Parameters

@dataclass
class Config:
    state_in:np.array
    objective:np.array
    dtype: tf.dtypes.DType
    num_layers: int
    steps: int
    optimizer: str = 'Adam'
    LR: float = 0.001
    LR_schedule: dict = field(default_factory=dict) # optional


class Circuit:
    def __init__(self, config:Config):
        self.config = config
        self.state_in = tf.cast(tf.constant(config.state_in), dtype=config.dtype)
        self.objective = tf.cast(tf.constant(config.objective), dtype=config.dtype)
        self._state_out = None
        self.cutoff = self.state_in.shape[0]
        self.optimizer = tf.optimizers.__dict__[config.optimizer](config.LR)
        self.parameters = Parameters(num_layers = self.config.num_layers, dtype=config.dtype)
        

    def _layer_out(self, gamma: tf.Tensor, phi: tf.Tensor, zeta: tf.Tensor, k: tf.Tensor, layer_in: tf.Tensor) -> tf.Tensor:
        layer_out = GaussianTransformation(gamma, phi, zeta, layer_in)
        return kerr(k, self.cutoff, self.config.dtype)*layer_out
    
    @property # lazy property
    def state_out(self) -> tf.Tensor:
        if self._state_out is None: 
            state = self.state_in
            for i in range(self.config.num_layers):
                state = self._layer_out(self.parameters.gamma[i], self.parameters.phi[i], self.parameters.zeta[i], self.parameters.kappa[i], state)
            self._state_out = state
        return self._state_out
    
    @property
    def fidelity(self) -> Union[tf.float64, tf.float32]:
        return tf.abs(tf.reduce_sum(self.state_out*tf.math.conj(self.objective)))**2

    def loss(self) -> Union[tf.float64, tf.float32]:
        return 1.0 - self.fidelity

    def minimize_step(self) -> float:
        self._state_out = None # reset lazy output state
        self.parameters.save() # save current values before updating
        self.optimizer.minimize(self.loss, self.parameters.learnable)
        return self.loss()
        

    def minimize(self) -> list:
        loss_list = []
        for i in tqdm(range(self.config.steps)):
            try:
                loss_list.append(self.minimize_step())

                # LR scheduling
                for threshold, new_LR in self.config.LR_schedule.items():
                    if loss_list[-1] < threshold:
                        self.optimizer.lr = new_LR

                if i%10 == 0:
                    print(f"Step {i}/{self.config.steps}: Fidelity = {100*(self.fidelity):.3f}%, LR = {self.optimizer.lr.numpy():.5f}", end='\r')

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"other exception: {e}")
                raise e
        return loss_list