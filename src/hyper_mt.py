import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from objective_wrapper import objective_wrapper

# Hyperopt commands
# mongod --dbpath . --port 1234 --directoryperdb --journal --nohttpinterface
# python hyper_mt.py
# hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1 --last-job-timeout 3



trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp3')

space = {'model_choice': 'unet-hypercol',
         'loss_choice': 'focalloss',
         'epochs': 1,
         'lr': 0.001,
         'batch_size': 16,
         'num_steps': 10,
         'prefix': None,
         'num_val_images': 2,
         'input_dir': '/data2/cgp/airbus_ships/',
         'min_delta': 20.0,
         'use_dropout_choice': False,
         'gamma': hp.uniform('gamma', 1.0, 2.0),
         'alpha': hp.uniform('alpha', 0.10, 0.25),
         'wandb_logging': False}

best = fmin(objective_wrapper,
    space=space,
    algo=tpe.suggest,
    max_evals=3,
    trials=trials)

print(best)
print(trials.losses())