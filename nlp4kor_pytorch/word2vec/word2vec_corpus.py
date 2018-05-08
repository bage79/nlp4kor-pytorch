import argparse
import codecs
import gzip
import os
import pickle
import random
import traceback

import numpy
from torch.utils.data import Dataset

from bage_utils.file_util import FileUtil
from bage_utils.num_util import NumUtil
from nlp4kor_pytorch.config import log, WIKIPEDIA_SENTENCE_FILE, WIKIPEDIA_DATA_DIR
from nlp4kor_pytorch.word2vec.word2vec_vocab import Word2VecVocab


class Word2VecCorpus(Dataset):
    WINDOW = 1
    SIDE = 'both'  # ['both', 'front', 'back']
    DEFAULT_FILE = f'{Word2VecVocab.DEFAULT_FILE}.window_{WINDOW}.side_{SIDE}.corpus'
    SAMPLE_FILE = f'{Word2VecVocab.SAMPLE_FILE}.window_{WINDOW}.side_{SIDE}.corpus'

    def __init__(self, data, vocab, window=WINDOW, side=SIDE):
        self.filepath = None
        self.ss_t = None
        self.vocab = vocab
        self.data = data
        self.window = window
        self.side = side

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = f'{self.__class__.__name__}({os.path.basename(self.filepath)}) len:{len(self)}'
        return s

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, numpy.array(owords)

    def set_subsampling(self, subsample_threshold=1e-5):
        if self.vocab.idx2freq is not None and subsample_threshold is not None:
            idx2freq = 1 - numpy.sqrt(subsample_threshold / self.vocab.idx2freq)
            idx2freq = numpy.clip(idx2freq, 0, 1)

            data_subsampled = []
            for iword, owords in self.data:
                if random.random() > idx2freq[iword]:
                    data_subsampled.append((iword, owords))
            self.data = data_subsampled

    @classmethod
    def skipgram(cls, sentence, i, window: int, side: str):
        iword = sentence[i]
        left = sentence[max(i - window, 0): i]
        right = sentence[i + 1: i + 1 + window]

        owords = []
        if side == 'both' or 'left':
            owords.extend([Word2VecVocab.UNK_CHAR for _ in range(window - len(left))] + left)
        if side == 'both' or 'right':
            owords.extend(right + [Word2VecVocab.UNK_CHAR for _ in range(window - len(right))])

        return iword, owords

    @classmethod
    def get_filepath(cls, data_dir, vocab, window, side):
        return os.path.join(data_dir, f"{os.path.basename(vocab.filepath)}.window_{window}.side_{side}.corpus")

    # noinspection PyAttributeOutsideInit
    @classmethod
    def build(cls, text_file: str, vocab: Word2VecVocab, window=5, side='both', data_dir=None) -> 'Word2VecCorpus':
        log.info(f"build corpus... {text_file}")
        if data_dir is None:
            data_dir = os.path.dirname(text_file)
        filepath = cls.get_filepath(data_dir=data_dir, vocab=vocab, window=window, side=side)

        if os.path.exists(filepath):
            log.info(f"corpus file exists. load {filepath}")
            return Word2VecCorpus.load(filepath)

        total_lines = FileUtil.count_lines(text_file)
        word2idx = {vocab.idx2word[idx]: idx for idx, _ in enumerate(vocab.idx2word)}
        data = []
        if text_file.endswith('.gz') or text_file.endswith('zip'):
            f = gzip.open(text_file, 'r')
        else:
            f = codecs.open(text_file, 'r', encoding='utf-8')
        with f:
            for no, line in enumerate(f):
                if no % 100000 == 0:
                    log.info(f"{os.path.basename(text_file)} {no/total_lines*100:.1f}% readed.")
                line = line.strip()
                if len(line) == 0:
                    continue
                sent = []
                for word in line.split():
                    if word in word2idx.keys():
                        sent.append(word)
                    else:
                        sent.append(Word2VecVocab.UNK_CHAR)
                for i in range(len(sent)):
                    iword, owords = cls.skipgram(sent, i, window=window, side=side)
                    data.append((word2idx[iword], [word2idx[oword] for oword in owords]))

        corpus = Word2VecCorpus(data=data, vocab=vocab, window=window, side=side)
        corpus.save(filepath=filepath)
        log.info(f"build corpus OK. {filepath}")
        return corpus

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            self.filepath = os.path.abspath(filepath)
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'Word2VecCorpus':
        with open(filepath, 'rb') as f:
            corpus = pickle.load(f)
            corpus.filepath = os.path.abspath(filepath)
            return corpus

    @property
    def data2text(self):
        for iword, owords in self.data:
            yield self.vocab.idx2word[iword], [self.vocab.idx2word[o] for o in owords]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', default=WIKIPEDIA_SENTENCE_FILE, type=str, help="corpus file path")
    parser.add_argument('--data_dir', default=WIKIPEDIA_DATA_DIR, type=str, help="data directory path (default:'./data')")

    parser.add_argument('--vocab_file', default=Word2VecVocab.DEFAULT_FILE, type=str)
    parser.add_argument('--window', default=Word2VecCorpus.WINDOW, type=int, help="window size")
    parser.add_argument('--side', default=Word2VecCorpus.SIDE, type=str, choices=['both', 'front', 'back'], help="target words in front or back or both (default: both)")
    args = parser.parse_args()
    try:
        log.info(f'vocab_file {args.vocab_file}')

        if not os.path.exists(args.vocab_file):
            log.error(f'vocab file does not exists. {args.vocab_file}')

        vocab = Word2VecVocab.load(args.vocab_file)
        log.info(vocab)
        for args.window in [args.window]:  # [1, 2, 3, 4, 5]:
            for args.side in [args.side]:  # ['both', 'front', 'back']:
                log.info(f'window: {args.window} side: {args.side}')
                corpus = Word2VecCorpus.build(text_file=args.text_file, vocab=vocab, window=args.window, side=args.side, data_dir=args.data_dir)
                log.info(f'corpus: {corpus.filepath} {NumUtil.comma_str(len(corpus))}')
    except:
        log.error(traceback.format_exc())
