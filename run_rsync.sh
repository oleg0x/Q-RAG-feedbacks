#!/bin/bash

rsync -ai --out-format="%n" \
    --exclude=.git \
    /DATA/Skoltech/Q-RAG-feedbacks \
    o.inozemcev@10.16.90.27:/home/o.inozemcev
