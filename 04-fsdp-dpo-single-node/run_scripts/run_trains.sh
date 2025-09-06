#!/bin/bash

accelerate launch \
	--num_processes 6 \
	-m "src.train_llm" \
	--length_strategy "regul"
