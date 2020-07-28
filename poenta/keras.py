import tensorflow as tf

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_modes: int = 1, dtype: tf.dtypes.DType = tf.complex64):
        super().__init__()
        self.num_modes = num_modes
        if dtype == tf.complex128:
            self.complextype = tf.complex128
            self.realtype = tf.float64
        elif dtype == tf.complex64:
            self.complextype = tf.complex64
            self.realtype = tf.float32
        else:
            raise ValueError("dtype can be only tf.complex64 or tf.complex128")

    def build(self, input_shape): # TODO: upgrade for 2 modes
        
        self.gamma = self.add_weight("gamma", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype))
        self.phi = self.add_weight("phi", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype))
        self.zeta = self.add_weight("zeta", dtype=self.complextype, trainable=True, initializer=complex_initializer(self.complextype))
        self.kappa = self.add_weight("kappa", dtype=self.realtype, trainable=True, initializer=real_initializer(self.realtype))
        self.cutoff = input_shape

    def call(self, input):
        gaussian_output = GaussianTransformation(self.gamma, self.phi, self.zeta, input)
        return kerr(self.kappa, self.cutoff[0], dtype=self.complextype) * gaussian_output



class QuantumDevice(tf.keras.Model):
  def __init__(self, num_layers, dtype):
    super().__init__(name='')
    self._layers = [QuantumLayer(dtype) for _ in range(num_layers)]

  def call(self, input_tensor):
    for layer in self._layers:
        input_tensor = layer(input_tensor)
    return input_tensor
