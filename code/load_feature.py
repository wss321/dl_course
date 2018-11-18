import pickle as pk
import numpy as np

feature_file = '../data/zero_shot_learning/tianchi_features/features.pkl'
cids_file = '../data/zero_shot_learning/tianchi_features/cids.pkl'
print('Load features data ... ')
feature = pk.load(open(feature_file, 'rb'))
cids = pk.load(open(cids_file, 'rb'))


def load_feature(train_cid, test_cid):
    train_feature = []
    test_feature = []
    train_label = []
    test_label = []
    for ft, cid in zip(feature, cids):
        if cid in train_cid:
            train_feature.append(ft)
            train_label.append(train_cid.index(cid))
        elif cid in test_cid:
            test_feature.append(ft)
            test_label.append(test_cid.index(cid))
    return np.asarray(train_feature), np.asarray(train_label), np.asarray(test_feature), np.asarray(test_label)


def load_feature_gzsl(train_cid, test_cid):
    train_feature = []
    test_feature = []
    train_label = []
    test_label = []
    for ft, cid in zip(feature, cids):
        if cid in train_cid:
            train_feature.append(ft)
            train_label.append((train_cid + test_cid).index(cid))
        elif cid in test_cid:
            test_feature.append(ft)
            test_label.append((train_cid + test_cid).index(cid))
    return np.asarray(train_feature), np.asarray(train_label), np.asarray(test_feature), np.asarray(test_label)


def load_all_feature(all_cid):
    all_feature = []
    all_label = []
    for ft, cid in zip(feature, cids):
        if cid in all_cid:
            all_feature.append(ft)
            all_label.append(all_cid.index(cid))
    return np.asarray(all_feature), np.asarray(all_label)
