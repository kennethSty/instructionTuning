#!/bin/bash

echo starting debug test
torchrun --nproc-per-node 2 -m debug.test_data_provider
