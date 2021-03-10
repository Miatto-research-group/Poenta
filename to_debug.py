from poenta.circuit import Circuit
from poenta.nputils import vaccum,hex_GKP,single_photon
import tensorflow as tf
import numpy as np

cutoff = 10
state_in = vaccum(1,cutoff)
# target_out = hex_GKP(1, 2, 0.3, 100, nmax=7)
target_out = single_photon(1,cutoff)
target_out2 = vaccum(1,cutoff)


device = Circuit(num_layers=5, num_modes=1, dtype=tf.complex128)

tuple_in_out = (state_in,target_out),(state_in,target_out2),
device.set_input_output_pairs(*tuple_in_out)
device.optimize(steps = 1000,optimizer = "SGD",learning_rate = 0.001,scheduler = False, nat_grad = True)