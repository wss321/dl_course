# Code to train the dem
import tensorflow as tf
import numpy as np
from kNN_cosine_or_euclideanl import kNNClassify
from word2vec_interface import find_word_vec
from attr_interface import find_attr_vec
import pickle
import time
from sklearn.metrics import accuracy_score
from train_test_classes import get_train_test_animal_classes
from load_feature import load_feature
import keras
from keras import Model
from keras.layers import Dense, Input, Activation

from keras import regularizers


def dem(input_shape, output_shape, norm_rate=0.0):
    inputs = Input(shape=(input_shape,), name='input')
    x = inputs
    x = Activation('relu')(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate), activation='relu')(x)
    predictions = Dense(output_shape, kernel_regularizer=regularizers.l2(norm_rate),
                        name='pred')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


meta_file = '../data/zero_shot_learning/tianchi_features/meta.pkl'
meta = pickle.load(open(meta_file, 'rb'))['class2cid']


def print_in_file(string, output_filename=None, verbose=True):
    """Prints a string into a file"""
    if output_filename is not None:
        output_file = open(output_filename, 'a')
        output_file.write(string + '\n')
        output_file.close()
    if verbose is True:
        time.sleep(0.01)
        print(string)
        time.sleep(0.01)


def read_pickle_file(filename):
    """Reads a pickle file using the latin1 encoding"""
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        p = u.load()
    return p


def spilt_class_file(file_path):
    return_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            return_data.append(line.split()[1])
    return return_data


def find_word(cid):
    for key, value in meta.items():
        if value == cid:
            return key
    return None


def find_cid(word):
    return meta[word]


def test_acc_fe(model):
    global test_id_per_class
    global test_feature
    global test_label
    global test_embeddings_per_class
    # shuffle_index = np.arange(test_feature.shape[0])
    # np.random.shuffle(shuffle_index)
    y_pred = model.predict(test_embeddings_per_class)
    if type(y_pred) == type(list()):
        y_pred = np.stack(y_pred, axis=1)
        y_pred = np.squeeze(y_pred)

    test_id_per_class = np.squeeze(np.asarray(test_id_per_class, dtype=np.int64))
    out_pred = np.zeros(test_feature.shape[0], dtype=np.int64)
    for i in range(test_feature.shape[0]):
        output_label = kNNClassify(test_feature[i, :], y_pred, test_id_per_class, 1)
        out_pred[i] = output_label

    accuracy = accuracy_score(out_pred, test_label.tolist())
    return accuracy


def train_acc_fe(model):
    global train_id_per_class
    global train_for_vali_feature
    global train_embedding_per_class
    global train_for_vali_label
    y_pred = model.predict(train_embedding_per_class)
    if type(y_pred) == type(list()):
        y_pred = np.stack(y_pred, axis=1)
        y_pred = np.squeeze(y_pred)
    train_id_per_class = np.squeeze(np.asarray(train_id_per_class, dtype=np.int64))
    out_pred = np.zeros(train_for_vali_feature.shape[0], dtype=np.int64)
    for i in range(train_for_vali_feature.shape[0]):
        output_label = kNNClassify(train_for_vali_feature[i, :], y_pred, train_id_per_class, 1)
        out_pred[i] = output_label

    accuracy = accuracy_score(out_pred, train_for_vali_label.tolist())
    return accuracy


class acc_callback_fe(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        test_accuracy = test_acc_fe(self.model)
        train_accuracy = train_acc_fe(self.model)
        print_in_file('train acc :{}\ttest acc:{}'.format(train_accuracy, test_accuracy),
                      '../result/model2_tianchi_{}.txt'.format(embed_type))


def __find_attr_word(classes):
    attr = find_attr_vec(classes)
    word = find_word_vec(classes)
    embedding = np.concatenate((attr, word), axis=0)
    return embedding


def get_embedding(classes, embed_type='train_word'):
    if embed_type == 'attr':
        fn = find_attr_vec
        size = 24
    elif embed_type == 'train_word':
        fn = find_word_vec
        size = 300
    else:
        fn = __find_attr_word
        size = 324
    embeddings = np.zeros(shape=(len(classes), size), dtype=np.float32)
    for index, cla in enumerate(classes):
        embeddings[index] = fn(cla)
    return embeddings


def train_test_split(features, label, embeddings_per_img, test_size=0.1, random_state=0, shuffle=True):
    np.random.seed(random_state)
    shuffle_index = np.arange(features.shape[0])
    if shuffle:
        np.random.shuffle(shuffle_index)
        print("shuffle_index:\n{}".format(shuffle_index))
    return features[shuffle_index[:int(features.shape[0] * (1 - test_size))]], \
           label[shuffle_index[:int(features.shape[0] * (1 - test_size))]], \
           features[shuffle_index[int(features.shape[0] * (1 - test_size)):]], \
           label[shuffle_index[int(features.shape[0] * (1 - test_size)):]], \
           embeddings_per_img[shuffle_index[:int(features.shape[0] * (1 - test_size))]], \
           embeddings_per_img[shuffle_index[int(features.shape[0] * (1 - test_size)):]]


if __name__ == '__main__':
    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    embed_type = 'attr'  # 'attr' 'train_word'

    # ----------------------------------------------------------
    # 加载数据
    train_class, test_class = get_train_test_animal_classes()
    train_cid_per_class = [find_cid(cla) for cla in train_class]
    test_cid_per_class = [find_cid(cla) for cla in test_class]

    train_feature, train_label, test_feature, test_label = load_feature(train_cid_per_class, test_cid_per_class)
    train_class_per_img = [find_word(train_cid_per_class[cid_index]) for cid_index in train_label]
    test_id_per_class = [test_cid_per_class.index(i) for i in test_cid_per_class]
    train_id_per_class = [train_cid_per_class.index(i) for i in train_cid_per_class]
    train_embeddings_per_img = get_embedding(train_class_per_img, embed_type)
    test_embeddings_per_class = get_embedding(test_class, embed_type)
    train_embedding_per_class = get_embedding(train_class, embed_type)

    print(train_feature.shape, train_label.shape, test_feature.shape, test_label.shape, train_embeddings_per_img.shape)
    print('LOAD DATA DONE.')
    # ----------------------------------------------------------
    from keras.optimizers import Adam

    model = dem(train_embeddings_per_img.shape[1], train_feature.shape[1], norm_rate=1e-5)

    loss_decay = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.5, min_lr=1e-7)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath='../zsl_model/model_{}.h5'.format(embed_type), monitor='loss',
                                                 verbose=1,
                                                 mode='auto',
                                                 save_best_only=True,
                                                 save_weights_only=False)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1,
                                                   mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir='../log')
    callback_lists = [early_stopping, checkpoint, loss_decay, tensorboard, acc_callback_fe()]
    model.compile(optimizer=Adam(1e-4), loss='cosine_proximity')  # mse
    model.summary()
    shuffle_index = np.arange(train_feature.shape[0])
    train_for_vali_feature = train_feature[shuffle_index[-1000:]]
    train_for_vali_label = train_label[shuffle_index[-1000:]]
    model.fit(x=train_embeddings_per_img[shuffle_index[:-1000]], y=train_feature[shuffle_index[:-1000]],
              batch_size=1024, epochs=100, callbacks=callback_lists,
              shuffle=True)
