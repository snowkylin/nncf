# coding = utf-8
import numpy as np
import csv
import networkx as nx


def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


class CiteULikeDataLoader:
    def __init__(self, data_dir='./data/citeulike/'):
        self.g = nx.DiGraph()
        self.user_num = 5551
        self.item_num = 16980
        self.word_id_dict = {}
        self.doc_title_list = []
        with open(data_dir + 'user-info.csv') as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                self.g.add_edge('u_' + str(int(row['user.id']) - 1), 'i_' + str(int(row['doc.id']) - 1))
        self.p_n = np.array([self.g.degree('i_%d' % i) for i in range(self.item_num)], dtype=np.float32)
        self.p_n = self.p_n / np.sum(self.p_n)
        with open(data_dir + 'raw-data.csv', encoding='utf-8', errors='ignore') as f:
            f_csv = csv.DictReader(f)
            word_id = 1
            for row in f_csv:
                word_list = row['title'].split(' ')
                for word in word_list:
                    if word not in self.word_id_dict:
                        self.word_id_dict[word] = word_id
                        word_id += 1
                self.doc_title_list.append([self.word_id_dict[word] for word in word_list])
        self.word_vector_dim = len(self.word_id_dict) + 1
        self.title_max_length = max([len(title) for title in self.doc_title_list])
        for title in self.doc_title_list:
            title.extend([0] * (self.title_max_length - len(title)))

    def fetch_batch(self, b=512, k=10, s=4, sampling_strategy='negative'):
        x_u_all = []
        x_v_all = []
        edges = self.g.edges()
        if sampling_strategy == 'negative':
            positive_links_list = [(int(edges[i][0][2:]), int(edges[i][1][2:]))
                                    for i in np.random.choice(self.g.number_of_edges(), size=b, replace=False)]
            for link in positive_links_list:
                x_u_all.append(link[0])
                x_v_all.append(self.doc_title_list[link[1]])
                for j in range(k):
                    while True:
                        negative_item_id = np.random.choice(self.item_num, p=self.p_n)
                        if negative_item_id != link[0]:
                            break
                    x_v_all.append(self.doc_title_list[negative_item_id])
        x_u_all = np.array(x_u_all)
        x_v_all = np.array(x_v_all)
        return one_hot_encode(x_u_all, self.user_num), one_hot_encode(x_v_all, self.word_vector_dim)
