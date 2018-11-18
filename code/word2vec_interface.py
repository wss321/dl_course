import numpy as np
from sklearn.decomposition import pca
import pandas as pd
from normalize import word2vec_norm_from_file

print('Loading train_word embeddings')
words = word2vec_norm_from_file('../data/zero_shot_learning/tianchi_features/class_wordembeddings.txt').set_index(0)
print('Loaded')
norm_mean = 1  # 5.5293


def find_norm_mean():
    """Find the mean norm of the word2vec representations"""
    all_words = words.index.values
    count = .0
    norm_sum = .0

    for w in all_words:
        new_norm = np.linalg.norm(words.loc[w].as_matrix())
        norm_sum += new_norm
        count += 1
    norm_sum /= count
    return norm_sum


def find_word_vec(word):
    """Gets the word2vec representation from a train_word"""
    try:
        return np.asarray(words.loc[word].as_matrix())
    except:
        return None


def do_pca(k=20):
    vec = []
    for word in words.index:
        vec.append(find_word_vec(word))
    pca_model = pca.PCA(n_components=k).fit(vec)
    Z = pca_model.transform(vec)
    Z = pd.DataFrame(Z, index=words.index)
    return Z


# def find_pca_word_vec(train_word, k=20):
#     """Gets the word2vec representation from a train_word"""
#     if k == 20:
#         Z = pca_20
#     elif k == 100:
#         Z = pca_100
#     elif k == 300:
#         Z = pca_300
#     else:
#         Z = do_pca(k=k)
#     try:
#         return Z.loc[train_word].as_matrix()
#     except:
#         return None

#
# pca_20 = do_pca(20)
# pca_100 = do_pca(100)
# pca_300 = do_pca(300)
