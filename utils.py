# coding = utf-8
import numpy as np
import csv
import networkx as nx
import pickle


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
    def __init__(self, data_dir='./data', test_data_rate=0.2):
        self.g_raw = nx.Graph()
        self.user_num = 5551
        self.item_num = 16980
        self.word_id_dict = {}
        self.doc_title_list = []

        # Build user-document (item) graph (info fetched from user-info.csv)

        with open(data_dir + '/citeulike/user-info.csv') as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                self.g_raw.add_edge('u_' + str(int(row['user.id']) - 1), 'i_' + str(int(row['doc.id']) - 1))

        # Fetch all titles of document (item) from raw-data.csv and translate words to word vectors

        self.word_vector_dict = pickle.load(open(data_dir + '/word_vectors/word_vector_dict.pkl', 'rb'))

        with open(data_dir + '/citeulike/raw-data.csv', encoding='utf-8', errors='ignore') as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                word_list = row['title'].split(' ')
                word_vector_list = []
                for word in word_list:
                    if word in self.word_vector_dict:       # Only add those words which are in the glove embedding file
                        word_vector_list.append(self.word_vector_dict[word])
                self.doc_title_list.append(word_vector_list)
        self.title_max_length = max([len(title) for title in self.doc_title_list])
        for title in self.doc_title_list:
            title.extend([[0] * 50] * (self.title_max_length - len(title)))

        # Select "test_data_rate" portion of items (documents) as test data and the others as train data

        self.all_users = ['u_' + str(i) for i in range(self.user_num)]
        self.train_items, self.test_items = [], []
        for i in range(self.item_num):
            if np.random.rand() < test_data_rate:
                self.test_items.append('i_' + str(i))
            else:
                self.train_items.append('i_' + str(i))
        self.g_train = self.g_raw.subgraph(self.all_users + self.train_items)
        self.g_test = self.g_raw.subgraph(self.all_users + self.test_items)

        # Calculate p_n of train data

        self.p_n = np.array([self.g_train.degree(i) for i in self.train_items], dtype=np.float32)
        self.p_n = self.p_n / np.sum(self.p_n)

        # Calculate p_d_user for all users (used in stratified sampling)

        self.p_d_user = np.array([self.g_train.degree(i) for i in self.all_users], dtype=np.float32)
        self.p_d_user = self.p_d_user / np.sum(self.p_d_user)
        self.p_d_item = self.p_n

    def fetch_batch(self, b=512, k=10, s=4, sampling_strategy='negative'):
        x_u_all = []
        x_v_all = []
        edges = self.g_train.edges()      # all positive links
        if sampling_strategy == 'negative':
            # Algorithm 2: Negative Sampling
            # Randomly draw b links from all positive links

            positive_links_list = [(int(edges[i][0][2:]), int(edges[i][1][2:])) if edges[i][0][0] == 'u'
                                    else (int(edges[i][1][2:]), int(edges[i][0][2:]))
                                    for i in np.random.choice(self.g_train.number_of_edges(), size=b, replace=False)]

            # For each positive link, draw k negative links

            for link in positive_links_list:
                x_u_all.append(link[0])
                x_v_all.append(self.doc_title_list[link[1]])
                for j in range(k):
                    while True:
                        negative_item_id = int(np.random.choice(self.train_items, p=self.p_n)[2:])
                        if negative_item_id != link[1]:
                            break
                    x_v_all.append(self.doc_title_list[negative_item_id])
        elif sampling_strategy == 'stratified_sampling':
            # Algorithm 3: Stratified Sampling

            item_list = np.random.choice(self.train_items, size=int(b / s), replace=False, p=self.p_d_item)

            for item in item_list:
                x_v_all.append(self.doc_title_list[int(item[2:])])
                users_list = [int(i[2:]) for i in np.random.choice(self.g_train.neighbors(item), size=s, replace=True)]
                # it is possible that s > #(neighbors)
                x_u_all.extend(users_list)

                for j in range(k * s):
                    while True:
                        negative_user_id = int(np.random.choice(self.all_users, p=self.p_d_user)[2:])
                        if negative_user_id not in users_list:
                            break
                    x_u_all.append(negative_user_id)
        elif sampling_strategy == 'negative_sharing':
            # Algorithm 4: Negative Sharing

            positive_links_list = [(int(edges[i][0][2:]), int(edges[i][1][2:])) if edges[i][0][0] == 'u'
                                   else (int(edges[i][1][2:]), int(edges[i][0][2:]))
                                   for i in np.random.choice(self.g_train.number_of_edges(), size=b, replace=False)]

            for link in positive_links_list:
                x_u_all.append(link[0])
                x_v_all.append(self.doc_title_list[link[1]])
        elif sampling_strategy == 'SS_with_NS':
            # Algorithm 5: Stratified Sampling with Negative Sharing

            item_list = np.random.choice(self.train_items, size=int(b / s), replace=False, p=self.p_d_item)

            for item in item_list:
                x_v_all.append(self.doc_title_list[int(item[2:])])
                users_list = [int(i[2:]) for i in np.random.choice(self.g_train.neighbors(item), size=s, replace=True)]
                # it is possible that s > #(neighbors)
                x_u_all.extend(users_list)

        x_u_all = np.array(x_u_all)
        x_v_all = np.array(x_v_all)
        return one_hot_encode(x_u_all, self.user_num), x_v_all

    def fetch_test_data(self):
        x_u_all = np.array(range(self.user_num))
        x_v_all = []
        for i in self.test_items:
            x_v_all.append(self.doc_title_list[int(i[2:])])
        return one_hot_encode(x_u_all, self.user_num), x_v_all

    def recall(self, r, top_n=50):
        pred = np.argsort(r)[:, -top_n:]
        res = 0.
        active_users = 0
        for u in self.all_users:
            pred_u = [int(self.test_items[i][2:]) for i in pred[int(u[2:])]]
            real_u = [int(i[2:]) for i in self.g_test.neighbors(u)]
            if len(real_u):
                active_users += 1
                res += len(set(pred_u) & set(real_u)) / len(real_u)
        return res / active_users
