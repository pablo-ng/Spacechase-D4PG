#!/bin/bash
source venv/bin/activate
jupyter lab&
tensorboard --logdir logs --reload_interval 5
# waiting for user interrupt
jupyter notebook stop 8888