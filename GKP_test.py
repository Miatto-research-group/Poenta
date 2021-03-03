import numpy as np
import time
from poenta.circuit import Circuit
from poenta.nputils import hex_GKP,vaccum
import tensorflow as tf
import logging

scheduler_status = False

logging.basicConfig(filename='GKP_test.log',level=logging.INFO,format='%(asctime)s %(message)s')
logging.info('Now start to traning~~~~')
logging.info("scheduler status is: %s",scheduler_status)

cutoff = 100
state_in = vaccum(1,cutoff)
target_out = hex_GKP(1, 2, 0.3, 100, nmax=7)


_time_min = 100000000
_loss_min = 100
for num_seed in np.arange(1,700,3):
    logging.info("number of seed is: %s",num_seed)
    device = Circuit(num_layers=25, num_modes=1, num_seed=num_seed, dtype=tf.complex128)
    tuple_in_out = (state_in,target_out),
    device.set_input_output_pairs(*tuple_in_out)
    start = time.time()
    device.optimize(steps = 5000,optimizer = "Adam",learning_rate = 0.001,scheduler = False, nat_grad = False)
    end = time.time()
    runtime_now = end-start
    loss_now = device._historycallback.losses[-1]
    if _time_min > runtime_now:
        _time_min = runtime_now
    if _loss_min > loss_now:
        _loss_min = loss_now
    logging.info("the runtime is: %f" ,runtime_now)
    logging.info("the fidelity is: %f", 1 - loss_now)
logging.info("========================")
logging.info("===time_min is %f====",_time_min)
logging.info("===highest_fidelity is %f====",_loss_min)
