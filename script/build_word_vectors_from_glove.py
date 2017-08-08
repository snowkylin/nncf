# coding = utf-8

import csv
import pickle

word_set = set()
with open('../data/citeulike/raw-data.csv', encoding='utf-8', errors='ignore') as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
        word_list = row['title'].split(' ')
        for word in word_list:
            word_set.add(word)

word_vector_dict = {}
with open('../data/word_vectors/glove.6b.50d.txt', encoding='utf-8') as f:
    for l in f.readlines():
        t = l.split(' ')
        if t[0] in word_set:
            word_vector_dict[t[0]] = [float(i) for i in t[1:]]

print(word_vector_dict)
print(len(word_vector_dict))
pickle.dump(word_vector_dict, open('../data/word_vectors/word_vector_dict.pkl', 'wb'))

