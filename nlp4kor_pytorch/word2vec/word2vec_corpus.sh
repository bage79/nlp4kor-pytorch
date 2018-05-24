#!/bin/bash -l
PROGRAM="word2vec_corpus.py"

echo "pkill -f ${PROGRAM}"
pkill -f "\./${PROGRAM}"
sleep 3

echo "rm -f logs/${PROGRAM}.*"
rm -f logs/${PROGRAM}.*

options=""
echo "python3 ./${PROGRAM} $options >/dev/null 2>&1 &"
nohup python3 ./${PROGRAM} $options >/dev/null 2>&1 &

