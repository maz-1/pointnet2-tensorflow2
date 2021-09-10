import os                            # filesystem operations
import numpy as np                   # maths
import tensorflow as tf              # neural networks
from tensorflow import keras         # neural networks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model

tf.random.set_seed(1234)
NUM_POINTS  = 2048
BATCH_SIZE  = 1
MODELPATH        = "/home/tomas/models/trained_pointnet2"

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

def create_dataset(train_points, test_points, train_labels, test_labels, batch_size = 16):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)

    return (train_dataset, test_dataset)

def build_model(config):
    if config['msg'] == True:
        model = CLS_MSG_Model(config['batch_size'], config['num_classes'], config['bn'])
    else:
        model = CLS_SSG_Model(config['batch_size'], config['num_classes'], config['bn'])

    model.build(input_shape=(config['batch_size'], NUM_POINTS, 3))
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

def train(model, train_dataset, test_dataset, epochs = 1):
    callbacks = [
        keras.callbacks.EarlyStopping(
            'val_sparse_categorical_accuracy', min_delta=0.01, patience=10),
        keras.callbacks.TensorBoard(
            './logs/{}'.format(config['log_dir']), update_freq=50),
        keras.callbacks.ModelCheckpoint(
            './logs/{}/model/weights.ckpt'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True)
    ]

    model.fit(
        train_dataset,
        validation_data = test_dataset,
        validation_steps = 20,
        validation_freq = 1,
        callbacks = callbacks,
        epochs = 1,
        verbose = 1,
        batch_size=config['batch_size']
    )

    return model


if __name__ == '__main__':

    # load in dataset
    dat = np.load("/home/tomas/datasets/ModelNet10.npz")
    train_points, test_points, train_labels, test_labels = dat["arr_0"], dat["arr_1"], dat["arr_2"], dat["arr_3"]
    CLASS_MAP = {0: 'desk', 1: 'bathtub', 2: 'night_stand', 3: 'bed', 4: 'dresser', 5: 'monitor', 6: 'toilet', 7: 'chair', 8: 'table', 9: 'sofa'}
    print("loaded")

    # make dataset
    train_set, test_set = create_dataset(train_points, test_points, train_labels, test_labels, batch_size=BATCH_SIZE)

    # build and train model
    config = {
        'train_ds' : 'data/modelnet_train.tfrecord',
        'val_ds' : 'data/modelnet_val.tfrecord',
        'log_dir' : 'msg_1',
        'batch_size' : BATCH_SIZE,
        'lr' : 0.001,
        'num_classes' : 10,
        'msg' : True, # multi-scale grouping model
        'bn' : False  # batch normalization
    }

    model = build_model(config)

    model = train(model, train_set, test_set)

    model.save(MODELPATH)