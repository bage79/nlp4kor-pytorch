import argparse
import codecs
import gzip
import os
import pickle
import traceback

import numpy

from bage_utils.file_util import FileUtil
from bage_utils.num_util import NumUtil
from nlp4kor_pytorch.config import log, WIKIPEDIA_SENTENCE_FILE, WIKIPEDIA_DATA_DIR, SAMPLE_WIKIPEDIA_SENTENCE_FILE


class Word2VecVocab(object):
    MAX_VOCAB = int(1e5)
    MIN_COUNT = 2
    TOKEN = 'word'  # ['word', 'morph', 'character', 'jaso']
    DEFAULT_FILE = os.path.join(WIKIPEDIA_DATA_DIR, f'{os.path.basename(WIKIPEDIA_SENTENCE_FILE)}.token_{TOKEN}.vocab_{MAX_VOCAB:.0e}.vocab')
    SAMPLE_FILE = os.path.join(WIKIPEDIA_DATA_DIR, f'{os.path.basename(SAMPLE_WIKIPEDIA_SENTENCE_FILE)}.token_{TOKEN}.vocab_{MAX_VOCAB:.0e}.vocab')

    UNK_IDX = 0
    UNK_CHAR = '¿'
    UNK_WORD = '¿'

    def __init__(self, token: str, min_count: int, idx2word: list, idx2freq: numpy.ndarray):
        self.filepath = None
        self.token = token
        self.min_count = min_count
        self.idx2word = idx2word
        self.idx2freq = idx2freq

    def __len__(self):
        return len(self.idx2word)

    def __repr__(self):
        s = f'{self.__class__.__name__}(len:{NumUtil.comma_str(len(self))}, min_count: {self.min_count}, {os.path.basename(self.filepath)})'
        return s

    @classmethod
    def get_filepath(cls, data_dir, text_file, vocab_size, token=TOKEN):
        return os.path.join(data_dir, f'{os.path.basename(text_file)}.token_{token}.vocab_{vocab_size:.0e}.vocab')

    @classmethod
    def build(cls, text_file: str, vocab_size=int(1e5), token=TOKEN, min_count=2, data_dir=WIKIPEDIA_DATA_DIR) -> 'Word2VecVocab':
        log.info(f"building vocab... {text_file}")
        if data_dir is None:
            data_dir = os.path.dirname(text_file)
        filepath = cls.get_filepath(data_dir, text_file, vocab_size)
        log.info(filepath)

        total_lines = FileUtil.count_lines(text_file)
        word2cnt = {}
        if text_file.endswith('.gz') or text_file.endswith('zip'):
            f = gzip.open(text_file, 'r')
        else:
            f = codecs.open(text_file, 'r', encoding='utf-8')
        with f:
            for no, line in enumerate(f):
                if no % 10000 == 0:
                    log.info(f"{os.path.basename(text_file)} {no/total_lines*100:.1f}% readed.")
                line = line.strip()
                if len(line) == 0:
                    continue
                sent = line.split()
                for word in sent:
                    word2cnt[word] = word2cnt.get(word, 0) + 1

        for word, cnt in word2cnt.copy().items():
            if cnt < min_count:
                del word2cnt[word]

        log.info(f'total unique words: {NumUtil.comma_str(len(word2cnt) + 1)}')
        idx2word = sorted(word2cnt, key=word2cnt.get, reverse=True)
        idx2word = [cls.UNK_CHAR] + idx2word[:vocab_size - 1]
        word2cnt[cls.UNK_CHAR] = 1
        idx2freq = numpy.array([word2cnt[word] for word in idx2word])
        idx2freq = idx2freq / idx2freq.sum()

        vocab = Word2VecVocab(token=token, min_count=min_count, idx2word=idx2word, idx2freq=idx2freq)
        vocab.save(filepath=filepath)
        log.info(f"build vocab OK. {filepath}")
        return vocab

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            self.filepath = os.path.abspath(filepath)
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'Word2VecVocab':
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
            vocab.filepath = os.path.abspath(filepath)
            return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', default=WIKIPEDIA_SENTENCE_FILE, type=str, help="corpus file path")
    parser.add_argument('--data_dir', default=WIKIPEDIA_DATA_DIR, type=str, help="data directory path (default:'./data')")

    parser.add_argument('--vocab_size', default=Word2VecVocab.MAX_VOCAB, type=int, help="maximum number of vocab (default:1e5)")
    parser.add_argument('--token', default=Word2VecVocab.TOKEN, choices=['word', 'morph', 'character', 'jaso'], help="token is word or morph or character (default: 'word')")
    parser.add_argument('--min_count', default=Word2VecVocab.MIN_COUNT, type=int)
    args = parser.parse_args()

    try:
        if not os.path.exists(args.text_file):
            log.error(f'text file does not exists. {args.text_file}')
            exit(-1)

        vocab = Word2VecVocab.build(text_file=args.text_file, vocab_size=args.vocab_size, token=args.token, min_count=args.min_count, data_dir=args.data_dir)
        log.info(f'vocab: {vocab.filepath} {NumUtil.comma_str(len(vocab))}')
        log.info(f'vocab.idx2word: {vocab.idx2word[:10]}')
        log.info(f'vocab.idx2freq: {vocab.idx2freq[:10]}')
    except:
        log.error(traceback.format_exc())
