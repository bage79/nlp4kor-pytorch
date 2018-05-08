import argparse
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore future warnings

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from bage_utils.tensorflow_util import TensorflowUtil
from nlp4kor_pytorch.config import TENSORBOARD_LOG_DIR, log
from nlp4kor_pytorch.word2vec.word2vec_embedding import Word2VecEmbedding

TensorflowUtil.turn_off_tensorflow_logging()


def word2vec_tensorboard(embedding_file_list, top_n=1e5, output_dir=TENSORBOARD_LOG_DIR):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for filename in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, filename))  # remove old tensorboard files

    config = projector.ProjectorConfig()
    embedding_list = []
    for embedding_file in embedding_file_list:
        if not os.path.exists(embedding_file):
            log.info(f'{embedding_file} not exists. skipped.')
            continue

        embedding = Word2VecEmbedding.load(embedding_file)

        name = os.path.basename(embedding_file.replace('+', ''))
        while name.startswith('_'):
            name = name[1:]

        idx2vec = embedding.idx2vec
        idx2word, idx2freq = embedding.idx2word, embedding.idx2freq
        if top_n > 0:
            name += f'.top_n_{top_n}'
            idx2vec, idx2word, idx2freq = idx2vec[:top_n], embedding.idx2word[:top_n], embedding.idx2freq[:top_n]

        embedding_var = tf.Variable(idx2vec, name=name)
        embedding_list.append(embedding_var)
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.join(output_dir, f'{name}.tsv')

        log.info('')
        log.info(f'{embedding_file} loaded.')
        log.info(f'embedding_var.name: {embedding_var.name} shape: {embedding_var.shape}')
        log.info(f'embedding.metadata_path: {embedding.metadata_path}')
        with open(embedding.metadata_path, 'wt') as out_f:
            out_f.write('spell\tfreq\n')
            for spell, freq in zip(idx2word, idx2freq):
                out_f.write(f'{spell}\t{freq:.7f}\n')

    summary_writer = tf.summary.FileWriter(output_dir)
    projector.visualize_embeddings(summary_writer, config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=embedding_list)
        checkpoint_file = os.path.join(output_dir, f'{name}.ckpt')
        saver.save(sess, checkpoint_file, global_step=None)
        log.info(f'checkpoint_file: {checkpoint_file}')

    # change absolute path -> relative path
    for filename in ['checkpoint', 'projector_config.pbtxt']:
        filepath = os.path.join(output_dir, filename)

        lines = []
        with open(filepath, 'rt') as f:
            for line in f.readlines():
                lines.append(line.replace(output_dir, '.'))
        os.remove(filepath)
        with open(filepath, 'wt') as f:
            for line in lines:
                f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard_dir', default=TENSORBOARD_LOG_DIR, type=str, help="data directory path")
    parser.add_argument('--embedding_file', default=Word2VecEmbedding.DEFAULT_FILE, type=str, help="word2vec embedding file")
    args = parser.parse_args()

    word2vec_tensorboard([args.embedding_file], top_n=int(1e5), output_dir=args.tensorboard_dir)
