#!/bin/bash

wget https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript ./data/
huggingface-cli download --local-dir ./data/pretrained_weights/wan2.1 alibaba-pai/Wan2.1-Fun-V1.1-14B-InP
