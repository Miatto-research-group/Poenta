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

import numpy as np
import tensorflow as tf
from poenta.circuit import Circuit

device = Circuit(num_layers=3, num_modes=1, dtype=tf.complex128)

state_in = tf.constant([1,0,0,0], dtype=tf.complex128)
target_out = tf.constant([0,1,0,0], dtype=tf.complex128)

def loss(target, output):
    return tf.abs(tf.reduce_sum(output * tf.math.conj(target))) ** 2

device.set_input_output_pairs([state_in,target_out])
device.optimize(loss, steps=100)

device.show_evolution(state_in)