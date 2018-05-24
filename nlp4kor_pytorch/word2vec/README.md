### Original
- https://github.com/theeluwin/pytorch-sgns

### download corpus
- 코퍼스의 크기 때문에 lfs file을 사용합니다.
- SourceTree 등의 tool을 이용하시면, lfs 파일의 git clone이 쉽습니다. 
```shell
cd ~/workspace
git clone https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org.git
```

### Create a vocabulary
```shell
./word2vec_vocab.sh
```
- create *.vocab file. (Word2VecVocab instance)
    - it takes 1~2 mins on macbook pro.
- vocab size: 100,000
- min count: 2
- unknown word: '¿'

### Create a corpus
```shell
./word2vec_corpus.sh
```
- create *.corpus file. (Word2VecCorpus instance)
    - it takes 5~10 mins on macbook pro.
- window size: 1

### Create a word2vec embedding
```shell
./word2vec_trainer.sh
```
- create *.embedding file. (Word2VecEmbedding instance)
    - it takes 16 hours on GPU PC. (GTX1080Ti)
- embedding size: 300 (bigger is better.)
- batch size: 500 (bigger is better.)
- negative samples: 100 (bigger is better.)
- optimizer & learning rate: Adam, 1e-4
- epoch: 20
- subsample rate: 1e-5

### Use the word2vec embedding
```shell
python ./word2vec_embedding_test.py
```
![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_embedding_test_1.png)

![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_embedding_test_2.png)

![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_embedding_test_3.png)


### Create tensorboard format files for visualization. (option)
- create checkpoint, *.ckpt, *.tsv, *.pbtxt files on tensorboard log directory.
    - it takes 10 ~ 20 secs on macbook pro.
```shell
./word2vec_tensorboard.sh
```
- Start tensorboard
```shell
tensorboard --logdir=~/tensorboard_log/ --port=6006
```

- http://localhost:6006/#projector
![screenshot](https://github.com/bage79/nlp4kor-pytorch/raw/master/ipynb/img/word2vec_tensorboard.png)
