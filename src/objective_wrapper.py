from hyperopt import fmin, tpe, hp, STATUS_OK
from run_training import run_training
import time


def objective_wrapper(params):
    summary = run_training(**params)
    res = {'loss': -1.0 * summary['max_fscore_bg'],
           'status': STATUS_OK,
           'eval_time': time.time()}
    return res