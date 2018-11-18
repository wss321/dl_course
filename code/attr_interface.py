import numpy as np
import pickle
from normalize import attr_normal_from_file

print('Loading attributes')
meta = pickle.load(open('../data/zero_shot_learning/tianchi_features/meta.pkl', 'rb'))
attrs = attr_normal_from_file('../data/zero_shot_learning/tianchi_features/attributes_per_class.txt').set_index(0)
# print(attrs)
print('Loaded')

norm_mean = 1  # 5.5293


def word_to_cid(word):
    class2cid = meta['class2cid']
    try:
        return class2cid[word]
    except:
        return None


def find_norm_mean():
    """Find the mean norm of the attributes representations"""
    all_attrs = attrs.index.values
    count = .0
    norm_sum = .0

    for a in all_attrs:
        new_norm = np.linalg.norm(attrs.loc[a].as_matrix())
        norm_sum += new_norm
        count += 1
    norm_sum /= count
    return norm_sum


def find_attr_vec(word):
    """Gets the attribute representation from a train_word"""
    try:
        cid = word_to_cid(word)
        return np.asarray(attrs.loc[cid].as_matrix() / norm_mean)
    except:
        return None
