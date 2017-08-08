import tensorflow as tf
import argparse
from model import NNCFModel
from utils import CiteULikeDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_vector_dim', default=50)
    parser.add_argument('--embedding_dim', default=50)
    parser.add_argument('--sampling_strategy', default='SS_with_NS',
                        help="negative, stratified_sampling, negative_sharing, SS_with_NS")
    parser.add_argument('--g_func', default='CNN', help="Item function (MoV and CNN)")
    parser.add_argument('--recall_top_n', default=50)
    parser.add_argument('--b', default=128)
    parser.add_argument('--k', default=10)
    parser.add_argument('--s', default=4)
    parser.add_argument('--learning_rate', default=1e-2)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_epoches', default=100000)
    parser.add_argument('--Lambda', default=1)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
    data_loader = CiteULikeDataLoader()
    args.seq_max_length = data_loader.title_max_length
    args.user_vector_dim = data_loader.user_num
    model = NNCFModel(args)
    x_u_test, x_v_test = data_loader.fetch_test_data()
    test_feed_dict = {model.x_u_all: x_u_test, model.x_v_all: x_v_test}
    print(args)
    print('batches\tloss\trecall')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for b in range(args.num_epoches):
            x_u_all, x_v_all = data_loader.fetch_batch(
                b=args.b, k=args.k, s=args.s, sampling_strategy=args.sampling_strategy
            )
            train_feed_dict = {model.x_u_all: x_u_all, model.x_v_all: x_v_all}
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=train_feed_dict)
            else:
                loss = sess.run(model.loss, feed_dict=train_feed_dict)
                r = sess.run(model.r_uv_all, feed_dict=test_feed_dict)
                recall = data_loader.recall(r, args.recall_top_n)
                print('%d\t%f\t%f' % (b, loss, recall))


def test(args):
    pass

if __name__ == '__main__':
    main()