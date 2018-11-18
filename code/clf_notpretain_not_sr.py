from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
import tensorflow as tf
import keras
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers.core import Dense
from keras import backend as K
from keras.applications.densenet import DenseNet121
import os
from keras.layers import regularizers
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.layers import PReLU

os.environ['KERAS_BACKEND'] = 'tensorflow'

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

POOLING = 'avg'
DATA_PATH = r'../data/classification/original'
MODEL_DIR = './clf_model'
TB_LOG = './log/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
CNN_MODEL_JSON = '{}/model.json'.format(MODEL_DIR)
SAVE_HISTORY_FILE = os.path.join(MODEL_DIR, 'train_history.txt')

"""# DenseNet
"""


def DenseNet(num_class=205, input_shape=(256, 256, 3), norm_rate=0.0, pooling='arg'):
    """重训练DenseNet"""
    base_model = DenseNet121(input_shape=input_shape, weights=None, include_top=False, pooling=pooling)
    x = base_model.output
    x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate), name='fc1')(x)
    x = PReLU()(x)
    predictions = Dense(num_class, activation='softmax', name='prediction')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def processing_img(img):
    def _add_random_noisy(img):
        rand = np.random.randint(100)
        if rand < 20:
            std = 3.0 / 255.0
            img = img + np.random.normal(0.0, std, img.shape)
            return img
        else:
            return img

    img = _add_random_noisy(img)
    rand = np.random.randint(100)
    if rand < 15:
        img = elastic_transform(img, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)
    return img


if __name__ == '__main__':

    train_path = DATA_PATH + '/train'
    vali_path = DATA_PATH + '/test'
    weight_file = 'weight.h5'
    train_batch_size = 32
    stop_patience = None
    best_monitor = 'val_loss'
    loss_decay_patience = 5
    classifier_init_lr = 1e-3
    epochs = 100000
    class_list = ['100_espresso', '101_alp', '102_cliff', '103_reef', '104_lakeside', '105_seashore', '106_salamander',
                  '107_acorn', '108_stork', '109_penguin', '10_mask', '110_albatross', '111_dugong', '112_submarine',
                  '113_platypus', '114_durian', '115_rickshaw', '116_hovercraft', '117_hedgehog', '118_terrier',
                  '119_freighter', '11_go-kart', '120_parrot', '121_tram', '122_turtle', '123_excavator', '124_carp',
                  '125_walnut', '126_hazelnut', '127_mangosteen', '128_boat', '129_retriever', '12_gondola',
                  '130_shepherd',
                  '131_bullfrog', '132_poodle', '133_tabby', '134_cat', '135_cougar', '136_lion', '137_bear',
                  '138_ladybug',
                  '139_fly', '13_centipede', '140_bee', '141_frog', '142_grasshopper', '143_stick', '144_cockroach',
                  '145_mantis', '146_dragonfly', '147_monarch', '148_butterfly', '149_cucumber', '14_hourglass',
                  '150_pig',
                  '151_hog', '152_alligator', '153_ox', '154_bison', '155_bighorn', '156_gazelle', '157_camel',
                  '158_orangutan', '159_chimpanzee', '15_ipod', '160_baboon', '161_elephant', '162_panda',
                  '163_constrictor', '164_abacus', '165_gown', '166_altar', '167_apron', '168_backpack',
                  '169_bannister',
                  '16_kimono', '170_barbershop', '171_barn', '172_barrel', '173_basketball', '174_trilobite',
                  '175_bathtub',
                  '176_wagon', '177_beacon', '178_beaker', '179_bikini', '17_lampshade', '180_binoculars',
                  '181_birdhouse',
                  '182_bowtie', '183_brass', '184_scorpion', '185_broom', '186_bucket', '187_train', '188_shop',
                  '189_candle', '18_mower', '190_cannon', '191_cardigan', '192_machine', '193_player', '194_chain',
                  '195_widow', '196_chest', '197_stocking', '198_dwelling', '199_keyboard', '19_lifeboat', '1_goldfish',
                  '200_confectionery', '201_convertible', '202_crane', '203_dam', '204_desk', '205_table',
                  '20_limousine',
                  '21_compass', '22_maypole', '23_goose', '24_uniform', '25_miniskirt', '26_van', '27_nail', '28_brace',
                  '29_obelisk', '2_tarantula', '30_oboe', '31_organ', '32_meter', '33_phone', '34_koala', '35_fence',
                  '36_bottle', '37_plunger', '38_pole', '39_poncho', '3_drumstick', '40_wheel', '41_projectile',
                  '42_bag',
                  '43_jellyfish', '44_reel', '45_refrigerator', '46_remote-control', '47_chair', '48_ball', '49_sandal',
                  '4_dumbbell', '50_bus', '51_scoreboard', '52_snorkel', '53_coral', '54_sock', '55_sombrero',
                  '56_heater',
                  '57_web', '58_sportscar', '59_stopwatch', '5_flagpole', '60_sunglasses', '61_bridge', '62_trunks',
                  '63_snail', '64_syringe', '65_teapot', '66_teddy', '67_thatch', '68_torch', '69_tractor',
                  '6_fountain',
                  '70_arch', '71_trolleybus', '72_turnstile', '73_umbrella', '74_vestment', '75_viaduct',
                  '76_volleyball',
                  '77_jug', '78_tower', '79_wok', '7_car', '80_spoon', '81_book', '82_plate', '83_guacamole', '84_slug',
                  '85_icecream', '86_lolly', '87_pretzel', '88_potato', '89_cauliflower', '8_pan', '90_pepper',
                  '91_mushroom', '92_orange', '93_lemon', '94_banana', '95_lobster', '96_pomegranate', '97_loaf',
                  '98_pizza', '99_potpie', '9_coat']
    train_list = class_list[:164]

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)
    K.set_session(session)
    tensorboard = TensorBoard(log_dir=TB_LOG)
    loss_decay = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=loss_decay_patience, verbose=1)

    checkpoint = ModelCheckpoint(filepath=weight_file, monitor=best_monitor, verbose=1,
                                 mode='auto',
                                 save_best_only=True,
                                 save_weights_only=True)
    if stop_patience is None:
        callback_lists = [checkpoint, loss_decay, tensorboard]
    else:
        early_stopping = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=1, mode='auto')
        callback_lists = [early_stopping, checkpoint, loss_decay, tensorboard]
    model = DenseNet(num_class=len(train_list), input_shape=(256, 256, 3), norm_rate=1e-4, pooling=POOLING)
    optm = SGD(classifier_init_lr, momentum=0.9)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    model_json = model.to_json()
    json_file = CNN_MODEL_JSON
    with open(json_file, "w") as j_f:
        j_f.write(model_json)
    model.summary()
    train_datagen = ImageDataGenerator(rotation_range=40.,
                                       width_shift_range=0.2,
                                       height_shift_range=0.15,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest',
                                       horizontal_flip=True,
                                       vertical_flip=False,
                                       preprocessing_function=processing_img,
                                       data_format='channels_last')
    test_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        train_path,
        classes=train_list,
        batch_size=train_batch_size,
        target_size=(256, 256))
    validation_generator = test_datagen.flow_from_directory(
        vali_path,
        classes=train_list,
        target_size=(256, 256),
        batch_size=32)
    h = model.fit_generator(
        generator=train_generator,
        verbose=1, steps_per_epoch=2000 / train_batch_size, epochs=epochs,
        callbacks=callback_lists,
        validation_data=validation_generator,
        validation_steps=300
    )
    with open(SAVE_HISTORY_FILE, 'a') as f:
        f.write('{}:{}\n'.format('BEST VAL_ACC', max(h.history['val_acc'])))
        for key, value in h.history.items():
            f.write('{}:{}\n'.format(key, str(value)))
        f.write('\n')
