import tensorflow as tf
import argparse
from model import NNCFModel
from utils import CiteULikeDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=10)
    parser.add_argument('--sampling_strategy', default='negative')
    parser.add_argument('--b', default=10)
    parser.add_argument('--k', default=10)
    parser.add_argument('--s', default=4)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_epoches', default=100000)
    parser.add_argument('--Lambda', default=128)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
    data_loader = CiteULikeDataLoader()
    args.word_vector_dim = data_loader.word_vector_dim
    args.seq_max_length = data_loader.title_max_length
    args.user_vector_dim = data_loader.user_num
    model = NNCFModel(args)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for b in range(args.num_epoches):
            x_u_all, x_v_all = data_loader.fetch_batch(
                b=args.b, k=args.k, s=args.s, sampling_strategy=args.sampling_strategy
            )
            feed_dict = {model.x_u_all: x_u_all, model.x_v_all: x_v_all}
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('batches %d, loss: %f' % (b, loss))


def test(args):
    pass

if __name__ == '__main__':
    main()