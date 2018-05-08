import argparse
import os
import sys
import traceback

import numpy
import torch
from torch.utils.data import DataLoader

from bage_utils.base_util import hostname, is_server
from bage_utils.date_util import DateUtil
from bage_utils.pytorch_util import PytorchUtil
from bage_utils.slack_util import SlackUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor_pytorch.config import log, WIKIPEDIA_SENTENCE_FILE, WIKIPEDIA_DATA_DIR
from nlp4kor_pytorch.word2vec.word2vec_corpus import Word2VecCorpus
from nlp4kor_pytorch.word2vec.word2vec_embedding import Word2VecEmbedding
from nlp4kor_pytorch.word2vec.word2vec_model import SGNSModel
from nlp4kor_pytorch.word2vec.word2vec_vocab import Word2VecVocab


class Word2VecTrainer(object):
    def __init__(self, vocab: Word2VecVocab, corpus: Word2VecCorpus, embed, neg_sample, neg_weight=True, subsample=None, batch=4096, learning_rate=1e-5, learning_decay=0.0, decay_start_epoch=1, device_no=None):
        self.embed = embed
        self.neg_sample = neg_sample
        self.neg_weight = neg_weight
        self.subsample = subsample
        self.batch = batch
        self.init_lr = learning_rate
        self.learning_decay = learning_decay
        self.decay_start_epoch = decay_start_epoch
        self.device_no = device_no

        self.window = corpus.window
        self.epoch = 0
        self.use_gpu = False if device_no is None else True
        # log.debug(f'batch: {batch}')

        if subsample is not None:
            corpus.set_subsampling(subsample_threshold=subsample)
        self.dataloader: DataLoader = DataLoader(corpus, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)
        # noinspection PyUnusedLocal

        self.sgns: SGNSModel = SGNSModel(vocab_size=len(vocab), embedding_size=embed, vocab=vocab, neg_sample=neg_sample, neg_weight=neg_weight)
        self.optim: torch.optim.Adam = torch.optim.Adam(self.sgns.parameters(), lr=learning_rate)
        if self.use_gpu and device_no is not None:
            PytorchUtil.use_gpu(device_no=device_no)
            # if len(device_no.split(',')) > 1:
            #     self.sgns = torch.nn.DataParallel(self.sgns).cuda()  # FIXME: not work
            self.sgns = self.sgns.cuda()

        # if self.learning_decay != 0:
        #     self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.learning_decay)
        #     # self.optimizer = PytorchUtil.exp_learing_rate_decay(self.optimizer, epoch=epoch, init_lr=self.learning_rate, lr_decay_epoch=1)
        # else:
        #     self.scheduler = None

    def __repr__(self):
        s = f'{self.__class__.__name__}(epoch:{self.epoch}, batch: {self.batch}, embed: {self.embed})'
        return s

    def train(self, iterations: int, batch: int, embedding: Word2VecEmbedding, args: argparse.Namespace) -> Word2VecEmbedding:
        batches_in_epoch = int(numpy.ceil(len(self.dataloader.dataset) / batch))
        total_batches = batches_in_epoch * iterations
        nth_total_batch = 0
        log.info(f'batches_in_epoch: {batches_in_epoch}')
        log.info(f'total_batches: {total_batches}')

        watch = WatchUtil(auto_stop=False)
        watch.start()
        best_loss = float("inf")
        first_epoch, last_epoch = self.epoch + 1, self.epoch + iterations + 1
        last_embedding_file = ''

        log.info(Word2VecEmbedding.get_filenpath(args))
        for self.epoch in range(first_epoch, last_epoch):
            log.info(f"[e{self.epoch:2d}] {self}")
            loss_list = []
            for nth, (iword, owords) in enumerate(self.dataloader, 1):
                try:
                    loss = self.sgns(iword, owords)
                except RuntimeError:
                    loss_list = [float('-inf')]
                    break

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # if nth_batch == 1 and self.scheduler is not None and self.epoch >= self.decay_start_epoch:  # TODO: TEST
                #     self.scheduler.step()

                if self.learning_decay != 0:
                    PytorchUtil.set_learning_rate(self.optim, self.epoch, gamma=self.learning_decay, base_lr=self.init_lr, min_lr=1e-10, decay_start=2, decay_interval=3)

                lr = PytorchUtil.get_learning_rate(self.optim)

                _, negatives = owords.size()
                real_loss = loss.data[0] / float(negatives)

                loss_list.append(real_loss)

                nth_total_batch += 1
                progressed = nth_total_batch / total_batches
                seconds_per_batch = float(watch.elapsed()) / float(nth_total_batch)
                remain_batches = total_batches - nth_total_batch
                remain_secs = int(seconds_per_batch * remain_batches)

                if nth == 1 or nth == batches_in_epoch or nth % 1000 == 0:
                    log.info(f"[e{self.epoch:2d}][b{nth:5d}/{batches_in_epoch:5d}][{progressed*100:.1f}% remain: {DateUtil.secs_to_string(remain_secs)}][window: {self.window}][lr: {lr:.0e}] loss: {real_loss:.7f}")

            total_loss = numpy.mean(loss_list)
            log.info(f"[e{self.epoch:2d}][window: {self.window}][lr: {lr:.0e}] total_loss: {total_loss:.7f}, best_loss: {best_loss:.7f}")
            if total_loss > best_loss or total_loss == float('inf') or total_loss == float('-inf'):  # bad loss than before or diverge
                log.info('')
                log.info(f"[e{self.epoch:2d}][window: {self.window}][lr: {lr:.0e}] total_loss > best_loss BREAK")
                log.info('')
                break
            else:
                best_loss = total_loss
                log.info(f"[e{self.epoch:2d}][window: {self.window}][lr: {lr:.0e}] embedding.save()...")
                args.epoch = self.epoch
                last_embedding_file = embedding.save(idx2vec=trainer.embedding, filepath=Word2VecEmbedding.get_filenpath(args))
                log.info(f"[e{self.epoch:2d}][window: {self.window}][lr: {lr:.0e}] embedding.save() OK. {os.path.basename(embedding.filepath)}")
        return last_embedding_file

    @property
    def embedding(self):
        return self.sgns.ivectors.weight.data.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_no', default="0" if torch.cuda.is_available() else None, type=str, help="use GPU or CPU (None: CPU)")

    parser.add_argument('--text_file', default=WIKIPEDIA_SENTENCE_FILE, type=str, help="corpus file path")
    parser.add_argument('--data_dir', default=WIKIPEDIA_DATA_DIR, type=str, help="data directory path (default:'./data')")
    # options for vocab, corpus

    parser.add_argument('--corpus_file', default=Word2VecCorpus.DEFAULT_FILE, type=str)

    # options for trains
    parser.add_argument('--epoch', default=Word2VecEmbedding.EPOCH, type=int, help="number of epochs (epoch=10 takes 4 hours) (default: 10)")
    parser.add_argument('--embed', default=Word2VecEmbedding.EMBED, type=int, help="embedding dimension (default: 300)")
    parser.add_argument('--batch', default=Word2VecEmbedding.BATCH, type=int, help="mini-batch size (default: 1e3)")  # batch가 작을 수록 loss가 매우 작아짐. batch 를 10분의 1로 줄이면, 학습시간은 2배로 걸림. (batch=1e4 -> mins/epoch=20)

    parser.add_argument('--neg_sample', default=Word2VecEmbedding.NEG_SAMPLE, type=int, help="number of negative samples (5~20 for small datasets, 2~5 for large datasets)")
    parser.add_argument('--neg_weight', default=Word2VecEmbedding.NEG_WEIGHT, action='store_true', help="use weights for negative sampling (None:same weights for all word")

    parser.add_argument('--subsample', default=Word2VecEmbedding.SUBSAMPLE, type=float, help="subsample threshold (default: 1e-5)")

    parser.add_argument('--learning_rate', default=Word2VecEmbedding.LEARNING_RATE, type=float, help="learning rate for AdamOptimizer")
    parser.add_argument('--learning_decay', default=Word2VecEmbedding.LEARNING_DECAY, type=float, help="exponential decay gamma (default: 0.0=no decay)")
    args = parser.parse_args()
    log.info(args)

    watch = WatchUtil(auto_stop=True)

    try:
        log.info(f'load {args.corpus_file} ...')
        watch.start()
        corpus = Word2VecCorpus.load(filepath=args.corpus_file)
        log.info(f'load {args.corpus_file} OK. (elapsed: {watch.elapsed_string()})')
        log.info(corpus.vocab)

        if len(corpus.vocab) > 1e5:  # out of memory (11GB GPU memory)
            args.device_no = None

        log.info('')
        log.info(args)
        log.info('')

        embedding_file = Word2VecEmbedding.get_filenpath(args)
        if os.path.exists(embedding_file):
            log.info(f'embedding_file: {embedding_file} exists. skipped')
            if is_server():
                SlackUtil.send_message(f'embedding_file: {embedding_file} exists. skipped')
                exit()

        log.info('')

        log.info(f'Word2VecTrainer() ...')
        watch.start()
        trainer = Word2VecTrainer(vocab=corpus.vocab, corpus=corpus, batch=args.batch, device_no=args.device_no,
                                  embed=args.embed, neg_sample=args.neg_sample, neg_weight=args.neg_weight, subsample=args.subsample,
                                  learning_rate=args.learning_rate, learning_decay=args.learning_decay)
        log.info(f'Word2VecTrainer() OK. (elapsed: {watch.elapsed_string()})')
        log.info(trainer)
        log.info(f'trainer.train(epoch={args.epoch}, batch={args.batch}) ...')
        watch.start()
        embedding = Word2VecEmbedding(filepath=embedding_file, vocab=corpus.vocab)
        embedding_file = trainer.train(iterations=args.epoch, batch=args.batch, embedding=embedding, args=args)
        log.info(f'embedding_file: {embedding_file} train OK. (elapsed: {watch.elapsed_string()})')

        if is_server():
            SlackUtil.send_message(f'embedding_file: {embedding_file} train OK. (elapsed: {watch.elapsed_string()})')
            SlackUtil.send_message(f'[{hostname()}][{args.device_no}] {sys.argv} OK.')
    except:
        log.error(traceback.format_exc())
        if is_server():
            SlackUtil.send_message(f'[{hostname()}][{args.device_no}] {sys.argv} ERROR.')
            SlackUtil.send_message(traceback.format_exc())
