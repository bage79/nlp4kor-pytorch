import logging
import os
import sys
import warnings

from bage_utils.base_util import is_server, db_hostname, is_pycharm_remote
from bage_utils.log_util import LogUtil

warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore future warnings

log = None
if log is None:
    if is_server():  # by shell script
        log = LogUtil.get_logger(sys.argv[0], level=logging.INFO, console_mode=True, multiprocess=False)  # global log
    elif is_pycharm_remote():  # remote gpu server
        log = LogUtil.get_logger(sys.argv[0], level=logging.DEBUG, console_mode=True, multiprocess=False)  # global log # console_mode=True for jupyter
    else:  # my macbook
        log = LogUtil.get_logger(None, level=logging.DEBUG, console_mode=True)  # global log

#################################################
# DB
#################################################
MONGO_URL = r'mongodb://%s:%s@%s:%s/%s?authMechanism=MONGODB-CR' % ('root', os.getenv('MONGODB_PASSWD'), 'db-local', '27017', 'admin')
MYSQL_URL = {'host': db_hostname(), 'user': 'root', 'passwd': os.getenv('MYSQL_PASSWD'), 'db': 'kr_nlp'}

#################################################
# tensorboard log dir
#################################################
TENSORBOARD_LOG_DIR = os.path.join(os.getenv("HOME"), 'tensorboard_log')
# log.info('TENSORBOARD_LOG_DIR: %s' % TENSORBOARD_LOG_DIR)
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)

#################################################
# nlp4kor-ko.wikipedia.org
#################################################
WIKIPEDIA_DIR = os.path.join(os.getenv('HOME'), 'workspace', 'nlp4kor-ko.wikipedia.org')
WIKIPEDIA_CORPUS_DIR = os.path.join(WIKIPEDIA_DIR, 'corpus')
if not os.path.exists(WIKIPEDIA_CORPUS_DIR):
    os.makedirs(WIKIPEDIA_CORPUS_DIR)

WIKIPEDIA_SENTENCE_FILE = os.path.join(WIKIPEDIA_CORPUS_DIR, 'ko.wikipedia.org.sentences')
SAMPLE_WIKIPEDIA_SENTENCE_FILE = os.path.join(WIKIPEDIA_CORPUS_DIR, 'sample.ko.wikipedia.org.sentences')

#################################################
# word2vec
#################################################
WORD2VEC_DATA_DIR = os.path.join(WIKIPEDIA_DIR, 'data', 'word2vec')
if not os.path.exists(WORD2VEC_DATA_DIR):
    os.makedirs(WORD2VEC_DATA_DIR)

WORD2VEC_EMBEDDING_FILE = os.path.join(WORD2VEC_DATA_DIR, 'ko.wikipedia.org.sentences.token_word.vocab_1e+05.vocab.window_1.side_both.corpus.embed_300.batch_500.neg_20.subsample_1e-05.lr_1e-03.decay_0.0.epoch_40.embedding')
