import tensorflow as tf
import numpy as np
from tqdm.notebook import tqdm
from dataclasses import dataclass, field
from Radicio import init_complex, init_real, GaussianTransformation, kerr

@dataclass
class Variables:
    gamma: tf.Variable
    phi: tf.Tensor
    zeta: tf.Variable
    kappa: tf.Variable
    
    @property
    def learnable(self):
        return [var for var in (self.gamma, self.phi, self.zeta, self.kappa) if isinstance(var, tf.Variable)]
    
    @property
    def all(self):
        return (self.gamma, self.phi, self.zeta, self.kappa)
    
    @property
    def L1_norm(self):
        tensor = tf.stack([tf.cast(var, tf.complex128) for var in self.all])
        return tf.abs(tf.linalg.norm(tensor, ord=1))

@dataclass
class Config:
    state_in:np.array
    objective:np.array
    dtype: tf.dtypes.DType
    num_layers: int
    steps: int
    optimizer: str = 'SGD'
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
        
        gamma = tf.Variable(init_complex(self.config.num_layers, 0.01), dtype=tf.complex128, name=f'gamma')
        phi = tf.constant(init_real(self.config.num_layers), dtype=tf.float64, name=f'phi')
        zeta = tf.Variable(init_complex(self.config.num_layers, 0.01), dtype=tf.complex128, name=f'zeta')
        kappa = tf.Variable(init_real(self.config.num_layers, 0.01), dtype=tf.float64, name=f'kappa')
        
        self.variables = Variables(gamma, phi, zeta, kappa)
        

    def _layer_out(self, gamma: tf.Tensor, phi: tf.Tensor, zeta: tf.Tensor, k: tf.Tensor, layer_in: tf.Tensor) -> tf.Tensor:
        layer_out = GaussianTransformation(gamma, phi, zeta, layer_in)
        return kerr(k, self.cutoff)*layer_out
    
    @property # lazy property
    def state_out(self) -> tf.Tensor:
        if self._state_out is None: 
            state = self.state_in
            for i in range(self.config.num_layers):
                state = self._layer_out(self.variables.gamma[i], self.variables.phi[i], self.variables.zeta[i], self.variables.kappa[i], state)
            self._state_out = state
        return self._state_out
    
    @property
    def fidelity(self) -> tf.float64:
        return tf.abs(tf.reduce_sum(self.state_out*tf.math.conj(self.objective)))**2

    def loss(self) -> tf.float64:
        return 1.0 - self.fidelity

    def minimize_step(self) -> float:
        self._state_out = None # reset lazy output state
        self.optimizer.minimize(self.loss, self.variables.learnable)
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
                    print(f"Fidelity = {100*(self.fidelity):.3f}%, LR = {self.optimizer.lr.numpy():.5f}", end='\r')

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"other exception: {e}")
                raise e
        return loss_list