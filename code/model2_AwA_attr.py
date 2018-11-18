import tensorflow as tf
import numpy as np
from kNN_cosine_or_euclideanl import kNNClassify
import time
from sklearn.metrics import accuracy_score
import keras
import scipy.io as sio
import h5py
from keras import Model
from keras.layers import Dense, Input, Activation

from keras import regularizers


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


def test_acc_fe(model):
    global x_test, test_id, test_label, att_pro
    y_pred = model.predict(att_pro)
    if type(y_pred) == type(list()):
        y_pred = np.stack(y_pred, axis=1)
        y_pred = np.squeeze(y_pred)

    test_id = np.squeeze(np.asarray(test_id, dtype=np.int64))
    out_pred = np.zeros(x_test.shape[0], dtype=np.int64)
    for i in range(x_test.shape[0]):
        output_label = kNNClassify(x_test[i, :], y_pred, test_id, 1)
        out_pred[i] = output_label
    accuracy = accuracy_score(out_pred, test_label.tolist())
    return accuracy


class acc_callback_fe(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        zsl_unseen_accuracy = test_acc_fe(self.model)
        print_in_file(
            'unseen acc:{}'.format(zsl_unseen_accuracy),
            '../result/model2_AwA_{}.txt'.format(embed_type))


def dem(input_shape, output_shape, norm_rate=0.0):
    inputs = Input(shape=(input_shape,), name='input')
    x = inputs
    x = Dense(300, kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    predictions = Dense(output_shape, kernel_regularizer=regularizers.l2(norm_rate), activation='relu',  #
                        name='pred')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


if __name__ == '__main__':
    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    embed_type = 'attr'
    # ----------------------------------------------------------
    # 加载数据
    f = h5py.File('../data/zero_shot_learning/AwA_data/attribute/Z_s_con.mat', 'r')
    att = np.array(f['Z_s_con'])
    print(att.shape)

    f = sio.loadmat('../data/zero_shot_learning/AwA_data/train_googlenet_bn.mat')
    x = np.array(f['train_googlenet_bn'])
    print(x.shape)

    f = sio.loadmat('../data/zero_shot_learning/AwA_data/test_googlenet_bn.mat')
    x_test = np.array(f['test_googlenet_bn'])
    print(x_test.shape)

    f = sio.loadmat('../data/zero_shot_learning/AwA_data/test_labels.mat')
    test_label = np.array(f['test_labels'])
    print(test_label.shape)

    f = sio.loadmat('../data/zero_shot_learning/AwA_data/testclasses_id.mat')
    test_id = np.array(f['testclasses_id'])
    print(test_id.shape)

    f = sio.loadmat('../data/zero_shot_learning/AwA_data/attribute/pca_te_con_10x85.mat')
    att_pro = np.array(f['pca_te_con_10x85'])
    print(att_pro.shape)

    f = sio.loadmat('../data/zero_shot_learning/AwA_data/wordvector/train_word.mat')
    train_word = np.array(f['train_word'])
    print('train_word', train_word.shape)

    f = sio.loadmat('../data/zero_shot_learning/AwA_data/wordvector/test_vectors.mat')
    test_word_per_class = np.array(f['test_vectors'])
    print('test_vectors', test_word_per_class.shape)

    print('LOAD DATA DONE.')
    # ----------------------------------------------------------
    from keras.optimizers import Adam

    model = dem(att.shape[1], x.shape[1], norm_rate=1e-4)

    loss_decay = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.5, min_lr=1e-7)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath='../zsl_model/model1_{}.h5'.format(embed_type), monitor='loss',
                                                 verbose=1,
                                                 mode='auto',
                                                 save_best_only=True,
                                                 save_weights_only=False)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1,
                                                   mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir='../log')
    callback_lists = [checkpoint, early_stopping, loss_decay, tensorboard, acc_callback_fe()]
    model.compile(optimizer=Adam(1e-4), loss='cosine_proximity')  # mse
    model.summary()
    model.fit(x=att, y=x, batch_size=1024, epochs=100000, callbacks=callback_lists,
              shuffle=True)
