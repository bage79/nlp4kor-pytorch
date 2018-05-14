import numpy
import torch

from bage_utils.num_util import NumUtil
from nlp4kor_pytorch.config import log


# noinspection PyUnresolvedReferences,PyArgumentList
class SGNSModel(torch.nn.Module):
    """
    original: https://github.com/theeluwin/pytorch-sgns
    """

    def __init__(self, vocab_size, embedding_size, vocab: 'Word2VecVocab', neg_sample=20, padding_idx=0, neg_weight=True):
        """

        :param word2vec:
        :param neg_sample: the number of negative sampling (5~20 for small datasets, 2~5 for large datasets)
        """
        super(SGNSModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = torch.nn.Parameter(
            torch.cat([torch.zeros(1, self.embedding_size), torch.FloatTensor(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = torch.nn.Parameter(
            torch.cat([torch.zeros(1, self.embedding_size), torch.FloatTensor(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

        self.vocab_size = len(vocab)
        self.neg_sample = neg_sample
        if (neg_weight and neg_sample > 0) and (vocab is not None and vocab.idx2freq is not None):
            self.ns_weights = numpy.power(vocab.idx2freq, 0.75)
            self.ns_weights = torch.FloatTensor(self.ns_weights / vocab.idx2freq.sum())
        else:
            self.ns_weights = None
        log.info(f'SGNSModel(vocab_size: {NumUtil.comma_str(self.vocab_size)}, embedding_size: {embedding_size}, neg_sample: {self.neg_sample})')

    def forward_i(self, data):
        v = torch.autograd.Variable(torch.LongTensor(data), requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = torch.autograd.Variable(torch.LongTensor(data), requires_grad=False)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ovectors(v)

    def forward(self, iword, owords):
        """
        C = context, N = negative samples, B = batch size, E = embedding size(dimension)
        :param iword:
        :param owords:
        :return:
        """
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.ns_weights is not None:
            nwords = torch.multinomial(self.ns_weights, batch_size * context_size * self.neg_sample, replacement=True).view(batch_size, -1)  # BCN -> Bx(CN)
        else:
            nwords = torch.FloatTensor(batch_size, context_size * self.neg_sample).uniform_(0, self.vocab_size - 1).long()  # Bx(CN)

        ivectors = self.forward_i(iword).unsqueeze(2)  # Bx(Ex1)
        ovectors = self.forward_o(owords)  # Bx(CxE)
        nvectors = self.forward_o(nwords).neg()  # Bx(CNxE)
        oloss = torch.bmm(ovectors, ivectors).sigmoid().log().squeeze().mean(dim=1)  # Bx(Cx1) -> mean -> B
        nloss = torch.bmm(nvectors, ivectors).sigmoid().log().view(-1, context_size, self.neg_sample).sum(dim=2).mean(dim=1)  # Bx(CN) -> Bx(CxN) -> sum -> Bx(C) -> mean -> B

        return -(oloss + nloss).clamp(-1e10, 1e10).mean()  # RuntimeError: value cannot be converted to type double without overflow: -inf
