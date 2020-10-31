import numpy as np
import tensorflow as tf
from poenta.circuit import Circuit
import time
import logging

scheduler_status = False

logging.basicConfig(filename='singlephoton_test.log',level=logging.INFO,format='%(asctime)s %(message)s')
logging.info('Now start to traning~~~~ with loss function: 1 - <out|target>')
logging.info("scheduler status is: %s",scheduler_status)

state_in = np.zeros(100, dtype=np.complex128)
state_in[0] = 1
target_out = np.zeros(100, dtype=np.complex128)
target_out[1] = 1

def loss_2(target, output):
    return  1 - tf.abs(tf.reduce_sum(output * tf.math.conj(target)))**2

def loss_1(target, output):
    return  1 - tf.abs(tf.reduce_sum(output * tf.math.conj(target_out)))
    

_time_min = 100
_loss_min = 100
for num_seed in np.arange(1,600,3):
    logging.info("number of seed is: %s",num_seed)
    device = Circuit(num_layers=8, num_modes=1, num_seed = num_seed, dtype=tf.complex128)
    tuple_in_out = (state_in,target_out),
    device.set_input_output_pairs(*tuple_in_out)
    start = time.time()
    device.optimize(loss_1, steps=1500, scheduler=scheduler_status)
    end = time.time()
    runtime_now = end-start
    loss_now = device._historycallback.losses[-1]
    if _time_min > runtime_now:
        _time_min = runtime_now
    if _loss_min > loss_now:
        _loss_min = loss_now
    logging.info("the runtime is: %f" ,runtime_now)
    logging.info("the loss is: %f", loss_now)
logging.info("========================")
logging.info("===time_min is %f====",_time_min)
logging.info("===loss_min is %f====",_loss_min)

