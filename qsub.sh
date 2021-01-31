#!/bin/bash

for pbs in "$@"
do
    qsub "$pbs"
done

