### Create a vocabulary
```shell
./word2vec_vocab.sh
```
- vocab size: 100,000
- min count: 2

### Create a corpus
```shell
./word2vec_corpus.sh
```
- window size: 1

### Create a word2vec embedding
```shell
./word2vec_trainer.sh
```
- batch size: 500
- epoch: 40
- subsample rate: 1e-5
- negative samples: 20
- embedding size: 300
- optimizer & learning rate: Adam, 1e-3

### Use the word2vec embedding
```shell
python ./word2vec_embedding_test.py
```
![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_embedding_test_1.png)

![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_embedding_test_2.png)

![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_embedding_test_3.png)


### Create tensorboard format files for visualization. (option)
```shell
./word2vec_tensorboard.sh
```

### Start tensorboard
```shell
tensorboard --logdir=~/tensorboard_log/ --port=6006
```
- http://localhost:6006/#projector
![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_tensorboard.png)
