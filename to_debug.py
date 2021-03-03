import numpy as np
import tensorflow as tf
from poenta.circuit import Circuit
from poenta.nputils import single_photon,NOON,hex_GKP
import matplotlib.pyplot as plt

state_in = np.zeros(50, dtype=np.complex128)
state_in[0] = 1
# target_out = hex_GKP(mu=1, d=2, delta=0.3, cutoff=50, nmax=7)
target_out = single_photon(50)

device = Circuit(num_layers=8, num_modes=1, dtype=tf.complex128)
tuple_in_out = (state_in,target_out),
device.set_input_output_pairs(*tuple_in_out)
device.optimize(steps=1000, optimizer="SGD", learning_rate=0.001, scheduler = False, nat_grad = False)

