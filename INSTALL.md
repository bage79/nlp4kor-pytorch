# nlp4kor (pytorch)
### Environment
- 서버: Ubuntu 16.04 + Anaconda3 4.5 + Python 3.6 + Pytorch 0.3
- 로컬: OSX 10.12 + Anaconda3 4.5 + Python 3.6 + Pytorch 0.3
- 환경변수 PYTHONPATH 에 `~/workspace/nlp4kor` 를 추가하시는 것이 좋습니다.
    - ~/.bash_profile 또는 ~/.profile 에 아래 한 줄을 추가해 주세요.
```shell
export PYTHONPATH=~/workspace/nlp4kor:$PYTHONPATH
```
    	
### Prerequisite
- python3.6 (https://www.anaconda.com/download/)
- tensorflow==1.5.0 (https://www.tensorflow.org/install/)
- tensorflow-tensorboard==1.5.1 (https://www.tensorflow.org/install/)
- torch==0.3.0.post4 (http://pytorch.org)
```shell
pip install --upgrade tensorflow==1.5.0
pip install --upgrade tensorflow-tensorboard==1.5.1
pip install --upgrade torch==0.3.0.post4
```

### download this source codes
```shell
mkdir ~/workspace
cd ~/workspace

pip install --upgrade pip

git clone https://github.com/bage79/nlp4kor.git
pip install -r ~/workspace/nlp4kor/requirements.txt

git clone https://github.com/bage79/nlp4kor-pytorch.git
pip install -r ~/workspace/nlp4kor-pytorch/requirements.txt
```

### install git & git lfs
- https://github.com/git-lfs/git-lfs/wiki/Installation
```shell
# OSX 기준 (최초 설치)
brew install git git-lfs
git lfs install
```

### download corpus
- 코퍼스의 크기 때문에 lfs file을 사용합니다.
- SourceTree 등의 tool을 이용하시면, lfs 파일의 git clone이 쉽습니다. 
```shell
cd ~/workspace
git clone https://gitlab.com/bage79/nlp4kor-ko.wikipedia.org.git
```
