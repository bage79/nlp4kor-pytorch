#!/bin/bash -l
PROGRAM="word2vec_tensorboard.py"

echo "pkill -f ${PROGRAM}"
pkill -f "\./${PROGRAM}"
sleep 3

echo "rm -f logs/${PROGRAM}.*"
rm -f logs/${PROGRAM}.*

option=""
echo "python3 ./${PROGRAM} $option >/dev/null 2>&1 &"
nohup python3 ./${PROGRAM} $option >/dev/null 2>&1 &
