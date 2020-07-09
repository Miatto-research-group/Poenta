import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from dataclasses import dataclass, field
from Radicio import init_complex, init_real, GaussianTransformation, kerr


class Variables:
    """
    Utility class to hold and manage the learnable and constant parameters of a circuit.
    And instance of this class is automatically created within an instance of the Circuit class.

    It allows to store, visualize and share optimization results.
    """

    def __init__(self, num_layers:int, dtype:tf.dtypes.DType = tf.complex64):
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
            raise ValueError(f'dtype can be only tf.complex128 or tf.complex64, not {dtype}')

        self.gamma = tf.Variable(init_complex(num_layers, 0.01), dtype=self.complextype, name=f'gamma')
        self.phi = tf.Variable(init_real(num_layers, 0.01), dtype=self.realtype, name=f'phi')
        self.zeta = tf.Variable(init_complex(num_layers, 0.01), dtype=self.complextype, name=f'zeta')
        self.kappa = tf.Variable(init_real(num_layers, 0.01), dtype=self.realtype, name=f'kappa')

        self._history:dict = {'gamma':[self.gamma.numpy()], 
                            'phi':[self.phi.numpy()],
                            'zeta':[self.zeta.numpy()],
                            'kappa':[self.kappa.numpy()]}

    @property
    def learnable(self):
        return [var for var in (self.gamma, self.phi, self.zeta, self.kappa) if isinstance(var, tf.Variable)]
    
    @property
    def all(self):
        return (self.gamma, self.phi, self.zeta, self.kappa)
    
    @property
    def L1_norm(self):
        tensor = tf.stack([tf.cast(var, self.complextype) for var in self.all])
        return tf.abs(tf.linalg.norm(tensor, ord=1))


    def plot(self, name:str, layer:int = None):
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
            values = values[:,layer]
        
        if name.lower() in {'gamma', 'zeta'}:
            fig, ax = plt.subplots(figsize=(7,7))
            ax.set_aspect('equal')

            xlim = 1.1*np.max(np.abs(np.real(values)))
            ylim = 1.1*np.max(np.abs(np.imag(values)))

            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel('Real')
            ax.set_ylabel('Imag')

            for v in np.transpose(values):
                plt.plot(np.real(v), np.imag(v))
                plt.grid()

        else:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.set_xlabel('step')
            ax.set_ylabel('value')
            
            for v in np.transpose(values):
                plt.plot(v)
            plt.grid()

        ax.set_title(name + ', all layers' if layer is None else f', layer {layer}')
        return ax

    def save(self):
        self._history['gamma'].append(self.gamma.numpy())
        self._history['phi'].append(self.phi.numpy())
        self._history['zeta'].append(self.zeta.numpy())
        self._history['kappa'].append(self.kappa.numpy())

    def __repr__(self):
        g = self.gamma.numpy()
        p = self.phi.numpy()
        z = self.zeta.numpy()
        k = self.kappa.numpy()
        string =f"gamma: {g}\nphi: {p}\nzeta: {z}\nkappa: {k}"
        return string
        

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
        self.variables = Variables(num_layers = self.config.num_layers, dtype=config.dtype)
        

    def _layer_out(self, gamma: tf.Tensor, phi: tf.Tensor, zeta: tf.Tensor, k: tf.Tensor, layer_in: tf.Tensor) -> tf.Tensor:
        layer_out = GaussianTransformation(gamma, phi, zeta, layer_in)
        return kerr(k, self.cutoff, self.config.dtype)*layer_out
    
    @property # lazy property
    def state_out(self) -> tf.Tensor:
        if self._state_out is None: 
            state = self.state_in
            for i in range(self.config.num_layers):
                state = self._layer_out(self.variables.gamma[i], self.variables.phi[i], self.variables.zeta[i], self.variables.kappa[i], state)
            self._state_out = state
        return self._state_out
    
    @property
    def fidelity(self) -> Union[tf.float64, tf.float32]:
        return tf.abs(tf.reduce_sum(self.state_out*tf.math.conj(self.objective)))**2

    def loss(self) -> Union[tf.float64, tf.float32]:
        return 1.0 - self.fidelity

    def minimize_step(self) -> float:
        self._state_out = None # reset lazy output state
        self.variables.save() # save current values before updating
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
                    print(f"Step {i}/{self.config.steps}: Fidelity = {100*(self.fidelity):.3f}%, LR = {self.optimizer.lr.numpy():.5f}", end='\r')

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"other exception: {e}")
                raise e
        return loss_list