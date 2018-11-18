from keras.models import model_from_json, Model
import tensorflow as tf
from keras import backend as K
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'


def model_from_json_and_weight(model_json_file, model_weight_file=None):
    # load json and create zsl_model
    print('Loading model from json:{} ...'.format(model_json_file))
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new zsl_model
    print('Done.')
    if model_weight_file is not None:
        print('Loading model weight from:{} ...'.format(model_weight_file))
        model.load_weights(model_weight_file)
        print('Done.')
    return model


def get_feature(model_json_file, model_weight_file, data, summary=False):
    """获取数据通过CNN后的属性"""
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    K.set_session(session)
    model = model_from_json_and_weight(model_json_file, model_weight_file)
    if summary:
        model.summary()
    top_layer = 'fc1'
    model = Model(inputs=model.input, outputs=model.get_layer(top_layer).output)
    feature = model.predict(data, verbose=1)
    K.clear_session()
    return feature


if __name__ == '__main__':
    import pickle as pk
    import numpy as np
    from keras.preprocessing.image import load_img, img_to_array
    from tqdm import tqdm

    np.random.seed(0)
    data_path = r'../data/classification/sr/'
    fname_file = r'../data/classification/sr/filenames.pkl'
    cid_file = r'../data/classification/sr/all_cids.pkl'
    file_names = pk.load(open(fname_file, 'rb'))
    file_names = np.asarray([data_path + fn for fn in file_names])
    cids = np.asarray(pk.load(open(cid_file, 'rb')))
    f_c = np.stack((file_names, cids), axis=1)
    np.random.shuffle(f_c)
    file_names = f_c[:, 0]
    cids = f_c[:, 1]
    print('file len:{}\t cid len:{}'.format(len(file_names), len(cids)))
    json = '../clf_model/model.json'
    weight = '../clf_model/weight.h5'
    batch_size = 10000
    data = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    features = np.zeros(shape=(len(cids), 1024), dtype=np.float32)
    for idx in range(int(len(file_names) / batch_size) + 1):
        start = idx * batch_size
        print('start with {}'.format(start))
        i = 0
        if int(len(file_names) / batch_size) == idx:
            for name in tqdm(file_names[start:], desc='fold-{} last'.format(idx + 1)):
                img = load_img(name)
                img = img_to_array(img)
                data[i] = img
                i += 1
            cid = cids[start:]
        else:
            for name in tqdm(file_names[start:start + batch_size], desc='fold-{}'.format(idx + 1)):
                img = load_img(name)
                img = img_to_array(img)
                data[i] = img
                i += 1
            cid = cids[start:start + i]
        feature = get_feature(json, weight, data[:i], summary=False)
        features[start:start + len(feature)] = feature
    pk.dump(features, open('../data//zero_shot_learning/tianchi_features/features.pkl', 'wb'))
    pk.dump(cids, open('../data//zero_shot_learning/tianchi_features/cids.pkl', 'wb'))
    print('Done.')
