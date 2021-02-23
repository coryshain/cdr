#!/bin/bash

for pbs in "$@"
do
    sbatch "$pbs"
done

