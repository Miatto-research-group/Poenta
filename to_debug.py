# import numpy as np
# import tensorflow as tf
# from poenta.keras import QuantumDevice

# device = QuantumDevice(num_modes=1, num_layers=3, cutoff=4, dtype=tf.complex128)

# state_in = tf.constant([1,0,0,0], dtype=tf.complex128)
# target_out = tf.constant([0,1,0,0], dtype=tf.complex128)

# def loss(target, output):
#     return tf.abs(tf.reduce_sum(output * tf.math.conj(target))) ** 2
    
# device.compile(optimizer='Adam', loss=loss)

# device.fit([state_in],[target_out])

# import numpy as np
# import tensorflow as tf
# from poenta.circuit import Circuit

# device = Circuit(num_layers=3, num_modes=1, dtype=tf.complex128)

# state_in = tf.constant([1,0,0,0], dtype=tf.complex128)
# target_out = tf.constant([0,1,0,0], dtype=tf.complex128)

# def loss(target, output):
#     return tf.abs(tf.reduce_sum(output * tf.math.conj(target))) ** 2

# device.set_input_output_pairs([state_in,target_out])
# device.optimize(loss, steps=100)

import numpy as np
import tensorflow as tf
from poenta.circuit import Circuit

state_in = np.zeros((10,10), dtype=np.complex128)
state_in[0,0] = 1
target_out = np.zeros((10,10), dtype=np.complex128)
target_out[0,5] = 1/np.sqrt(2)
target_out[5,0] = 1/np.sqrt(2)

def loss(target, output):
    return 1 - tf.abs(tf.reduce_sum(output * tf.math.conj(target))) ** 2

for seed in [112,113,667,668]:
    device = Circuit(num_layers=20, num_modes=2, num_seed = seed, dtype=tf.complex128)
    tuple_in_out = (state_in,target_out),
    device.set_input_output_pairs(*tuple_in_out)
    device.optimize(loss, steps=50)

device.show_evolution(state_in)