#!/bin/bash

srun --exclusive --gres=gpu:4 \
	./main -n 100 -v $@