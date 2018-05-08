import argparse
import os
import pickle

import numpy
from scipy.spatial import distance

from nlp4kor_pytorch.word2vec.word2vec_corpus import Word2VecCorpus


class Word2VecEmbedding(object):
    BATCH = 200
    EPOCH = 40
    SUBSAMPLE = 1e-5
    NEG_SAMPLE = 20
    NEG_WEIGHT = True  # True in [True, False]
    EMBED = 300  # 300 is best in [50, 300]

    LEARNING_RATE = 1e-3  # 1e-3 is best in [1e-3, 1e-5]
    LEARNING_DECAY = 0.0

    DEFAULT_FILE = f'{Word2VecCorpus.DEFAULT_FILE}.embed_{EMBED}.batch_{BATCH}.neg_{NEG_SAMPLE}.subsample_{SUBSAMPLE:.0e}.lr_{LEARNING_RATE:.0e}.decay_{LEARNING_DECAY:.1f}.epoch_{EPOCH}.embedding'

    # DEFAULT_FILE = WORD2VEC_EMBEDDING_FILE

    # noinspection PyUnresolvedReferences
    def __init__(self, filepath: str, vocab: 'Word2VecVocab'):
        self.filepath = filepath
        self.vocab = vocab
        self.idx2vec = None
        self.idx2freq = None  # from vocab
        self.idx2word = None  # from vocab

        self.word2idx = None  # create while loading

    def __repr__(self):
        s = f'{self.__class__.__name__}({os.path.basename(self.filepath)}) len:{len(self.word2idx)}, array:{self.idx2vec.shape}'
        return s

    def save(self, idx2vec: numpy.ndarray = None, filepath: str = None) -> str:
        """
        save word2vec numpy array & vocab
        :param idx2vec: numpy array from Word2VecTrainer.train()
        :param filepath: .embedding file path
        :return:
        """
        if filepath is not None:
            self.filepath = filepath

        if not os.path.isdir(os.path.dirname(self.filepath)):
            os.mkdir(os.path.dirname(self.filepath))

        self.idx2word: list = self.vocab.idx2word
        self.idx2freq: numpy.ndarray = self.vocab.idx2freq
        if idx2vec is not None:
            self.idx2vec = idx2vec

        with open(self.filepath, 'wb') as f:
            pickle.dump(self, f)
        return self.filepath

    @classmethod
    def load(cls, filepath) -> 'Word2VecEmbedding':
        """
        load word2vec numpy array & vocab
        :param filepath:
        :return:
        """
        with open(filepath, 'rb') as f:
            embedding = pickle.load(f)
            embedding.word2idx = {spell: idx for idx, spell in enumerate(embedding.vocab.idx2word)}
            return embedding

    # noinspection PyUnresolvedReferences
    @classmethod
    def get_filenpath(cls, args: argparse.Namespace):
        filename = f'{os.path.basename(args.corpus_file)}.embed_{args.embed:d}.batch_{args.batch:d}.neg_{args.neg_sample:d}.subsample_{args.subsample}.lr_{args.learning_rate:.0e}.decay_{args.learning_decay:.1f}.epoch_{args.epoch}.embedding'  # .neg_weight_{args.neg_weight}'
        return os.path.join(args.data_dir, filename)

    def __len__(self):
        return len(self.idx2vec)

    def __getitem__(self, word: str) -> numpy.ndarray:
        return self.idx2vec[self.word2idx.get(word, 0)]

    def __contains__(self, word: str):
        return True if word in self.word2idx else False

    def __iter__(self):
        for word in self.word2idx.keys():
            yield word

    def freq(self, word: str) -> float:
        """
        frequent of word
        :param word:
        :return:
        """
        return self.idx2freq[self.word2idx.get(word, 0)]

    def most_frequent(self, top_n=100) -> [str]:
        """
        most frequent words
        :param top_n: top N
        :return:
        """
        return [self.idx2word[idx] for idx in range(1, top_n + 1)]

    def similarity(self, word1: str, word2: str, metric='cosine') -> float:
        """
        similarity between two words
        :param word1:
        :param word2:
        :param metric:
        :return:
        """
        if 0 == self.word2idx.get(word1, 0) or 0 == self.word2idx.get(word2, 0):
            return 0.

        return self.similarity_vec(self[word1], self[word2], metric=metric)
        # vec1 = self.__getitem__(word1).reshape((1, -1))
        # vec2 = self.__getitem__(word2).reshape((1, -1))
        # return 1 - distance.cdist(vec1, vec2, metric=metric).reshape(-1)

    # noinspection PyMethodMayBeStatic
    def similarity_vec(self, vec1: numpy.ndarray, vec2: numpy.ndarray, metric='cosine') -> float:
        """
        similarity between two words
        :param vec1:
        :param vec2:
        :param metric: 'cosine' or 'euclidean'
        :return:
        """
        if numpy.count_nonzero(vec1) == 0 or numpy.count_nonzero(vec2) == 0:
            if metric == 'cosine':
                return 0.
            else:
                return 0.

        vec1 = vec1.reshape((1, -1))
        vec2 = vec2.reshape((1, -1))
        if metric == 'cosine':
            return (1 - distance.cdist(vec1, vec2, metric=metric).reshape(-1))[0]
        else:
            return distance.cdist(vec1, vec2, metric=metric).reshape(-1)[0]

    def most_similar_vec(self, vec: numpy.ndarray, top_n=3, exclude_words=[], metric='cosine') -> [(str, float)]:
        vec = vec.reshape((1, -1))
        mat = self.idx2vec[1:, :]
        if metric == 'cosine':
            sims = 1 - distance.cdist(mat, vec, metric).reshape(-1)
            idxs = sims.argsort()[::-1] + 1
            sims = numpy.sort(sims)[::-1][:top_n + len(exclude_words)].tolist()
        else:
            sims = distance.cdist(mat, vec, metric).reshape(-1)
            idxs = sims.argsort() + 1
            sims = numpy.sort(sims)[:top_n + len(exclude_words)].tolist()

        # print('len(words):', len(exclude_words))
        words = [self.idx2word[idx] for idx in idxs[:top_n + len(exclude_words)]]
        # print('sims:', sims)
        # print('words:', words)
        if len(exclude_words) == 0:
            return [(w, sim) for w, sim in zip(words, sims)][:top_n]
        else:
            return [(w, sim) for w, sim in zip(words, sims) if w not in exclude_words][:top_n]

    # noinspection PyDefaultArgument
    def most_similar(self, words: [str], top_n=3, metric='cosine') -> [(str, float)]:
        """
        find most similar words
        :param words:
        :param top_n: number of data, default: 3
        :param metric: 'cosine' or 'euclidean', defulat: 'cosine'
        :return:
        """
        if len(words) == 0:
            return []

        vec = self.mean(words)
        if numpy.count_nonzero(vec) == 0:
            return []

        return [w for w, sim in self.most_similar_vec(vec=vec, top_n=top_n, exclude_words=words, metric=metric)]

    def relation_ab2xy(self, a: str, b: str, x: str, top_n=3, metric='cosine') -> [(str, float)]:
        """
        find y when a:b = x:y ('왕':'여왕' = '남자':Y)
        :param a:
        :param b:
        :param x:
        :param top_n: number of data, default: 3
        :param metric: 'cosine' or 'euclidean', defulat: 'cosine'
        :return:
        """
        if 0 == self.word2idx.get(a, 0) or 0 == self.word2idx.get(b, 0) or 0 == self.word2idx.get(x, 0):
            return None

        _a = self[a]
        _b = self[b]
        _x = self[x]
        _y = _b - _a + _x
        return ' '.join([y for y, sim in self.most_similar_vec(vec=_y, top_n=top_n, exclude_words=[a, b, x], metric=metric)])

    # noinspection PyMethodMayBeStatic
    def doesnt_match(self, words: [str], top_n=1, metric='cosine') -> str:
        if len(words) > 31:
            words = words[:31]

        # combi = [words for _ in range(len(words))]
        # for x in NumpyUtil.combinations(combi):
        #     print(x)
        sim_list = []
        for word in words:
            if word not in self:
                raise Exception(f'"{word}" not in embedding.')
            subwords = set(words) - {word}
            sim = self.similarity_vec(self[word], self.mean(subwords), metric=metric)
            # print(sim, word, subwords)
            sim_list.append((word, sim))

        sim_list = sorted(sim_list, key=lambda x: x[1], reverse=False)
        return ' '.join([w for w, sim in sim_list[:top_n]])

    def mean(self, words: [str]):
        vecs = numpy.array([self[word] for word in words])
        return numpy.mean(vecs, axis=0)

    def add_suffix(self, root: str, suffix: str, top_n=10, metric='cosine', nearest=100):
        y_list = []  # not found
        if root + suffix not in self:
            return y_list

        sim_words = [w for w in self.most_similar([root], top_n=nearest)]
        # print(root, sim_words)

        a_words, b_words = [], []
        for a in sim_words:
            if a in self and a + suffix in self:
                # print(a, a + suffix)
                a_words.append(a)
                b_words.append(a + suffix)

        # print(root, a_words, b_words)
        if len(a_words) > top_n and len(b_words) > top_n:
            a = self.mean(a_words)
            b = self.mean(b_words)
            if numpy.count_nonzero(a) == 0 or numpy.count_nonzero(b) == 0 or numpy.count_nonzero(self[root]) == 0:
                return y_list
            else:
                return [w for w, sim in self.most_similar_vec(b - a + self[root], top_n=top_n, exclude_words=[root] + a_words + b_words, metric=metric)[:top_n]]
        else:
            return y_list

    def roots(self, word: str, top_n=1, sort_by_len=False, score_with_preserve=False, min_sim=0.3, more_freq=False, metric='cosine') -> list:
        candi_list = []
        for a_len in range(1, len(word)):
            a = word[:a_len]
            b = word[a_len:]
            if a in self and b in self:
                for x in [a]:  # [a, b]:
                    if more_freq and self.freq(x) < self.freq(word):
                        continue

                    # if word not in self.add_suffix(root=a, suffix=b, top_n=100):
                    #     continue

                    sim = self.similarity(word, x, metric=metric)  # full vs root
                    if sim > min_sim:
                        if score_with_preserve:
                            preserve = self.similarity_vec(self[x] + self[b], self[word], metric=metric)  # full vs root + suffix
                            # print(x, b, 'x=word', sim, 'x+b=word', preserve, preserve - sim)
                            candi_list.append((x, preserve - sim))
                        else:
                            candi_list.append((x, sim))

        if len(candi_list) > 0:
            if sort_by_len:
                candi_list = sorted(candi_list, key=lambda x: len(x[0]), reverse=True)  # by word len
            else:
                candi_list = sorted(candi_list, key=lambda x: x[1], reverse=True)  # by sim
            return [root for root, _sim in candi_list[:top_n]]
        else:
            return []

    def importances(self, words: [str], metric='cosine'):
        if len(words) < 3:
            return [1.] * len(words)

        origin_vec = self.mean(words)

        result = []
        for i in range(len(words)):
            summary_vec = self.mean(words[:i] + words[i + 1:])
            sentence_sim = self.similarity_vec(origin_vec, summary_vec, metric=metric)
            # print(sentence_sim, candi_words[i])
            result.append(1. - sentence_sim)

        return result
