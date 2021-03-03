import numpy as np
import time
from poenta.circuit import Circuit
from poenta.nputils import single_photon,vaccum
import tensorflow as tf

cutoff = 100
state_in = vaccum(1,cutoff)
target_out = single_photon(1,cutoff)

device = Circuit(num_layers=8, num_modes=1, num_seed=11, dtype=tf.complex128)

tuple_in_out = (state_in,target_out),
device.set_input_output_pairs(*tuple_in_out)
start = time.time()
device.optimize(steps = 1500,optimizer = "Adam",learning_rate = 0.001,scheduler = False, nat_grad = False)
end = time.time()
runtime_now = end-start
print("the fidelity is: %f" , 1-device._historycallback.losses[-1])
print("the runtime is: %f" ,runtime_now)
    

