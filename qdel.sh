#!/bin/bash

i=$1

while [ $i -le $2 ]
do
    qdel $i
    i=`expr $i + 1`
done

