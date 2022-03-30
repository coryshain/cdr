#!/bin/bash

i=$1

while [ $i -le $2 ]
do
    scancel $i
    i=`expr $i + 1`
done

