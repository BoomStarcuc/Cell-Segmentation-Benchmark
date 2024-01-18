import argparse
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import SGD
from tensorflow.python.data import Dataset
from tqdm import tqdm

from deepcell import image_generators, losses, model_zoo
from deepcell.image_generators import ImageFullyConvDataGenerator
from deepcell.utils import train_utils, tracking_utils
from deepcell.utils.train_utils import count_gpus, get_callbacks, rate_scheduler

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FeatureNet Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default='path/to/your/dataset/directory', 
                        help='path to the data')
    parser.add_argument('--exp-name', type=str, default='featurenet_train_on_all_types_nuclear_1C', 
                        help='name of the experiment')
    parser.add_argument('--model-name', type=str, default='featurenet', 
                        help='name of the experiment')
    parser.add_argument('--log-dir', type=str, default='./featurenet_logs', 
                        help='where you want to put your logs')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Start training with existing weights')
    parser.add_argument('--chooseone', default='All',
                        help='choose from Tonsil Breast BV2 Epidermis SHSY5Y Esophagus \
                        A172 SkBr3 Lymph_Node Lung SKOV3 Pancreas Colon MCF7 Huh7 BT474 lymph node metastasis Spleen')
    args = parser.parse_args()

    # create folder for this set of experiments
    experiment_folder = args.exp_name
    MODEL_DIR = os.path.join("./FeatureNet", experiment_folder)

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    seed = args.seed
    data_dir = args.data_dir
    val_dict = np.load(os.path.join(data_dir, 'tissuenet_val_split_256x256_memserpreprocess.npz'), allow_pickle=True)
    train_dict = np.load(os.path.join(data_dir, 'tissuenet_train_split_256x256_memserpreprocess.npz'), allow_pickle=True)

    X_train = train_dict['X'][...,0].squeeze()
    y_train = train_dict['y'][...,1].squeeze()

    X_val = val_dict['X'][...,0].squeeze()
    y_val = val_dict['y'][...,1].squeeze()

    tissue_type_train = train_dict['tissue_list']
    tissue_type_val = val_dict['tissue_list']
    y_train = y_train.astype('int32').squeeze()
    y_val = y_val.astype('int32').squeeze()
        
    X_train = X_train
    y_train = y_train[..., np.newaxis]
    X_val = X_val
    y_val = y_val[..., np.newaxis]

    norm_method = 'std'  # data normalization
    receptive_field = 61  # should be adjusted for the scale of the data
    n_skips = 1  # number of skip-connections (only for FC training)
    dilation_radius = 1  # change dilation radius for edge dilation
    separate_edge_classes = True  # break edges into cell-background edge, cell-cell edge
    pixelwise_kwargs = {
        'dilation_radius': dilation_radius,
        'separate_edge_classes': separate_edge_classes,
    }

    fgbg_model = model_zoo.bn_feature_net_skip_2D(
        n_features=2,  # segmentation mask (is_cell, is_not_cell)
        receptive_field=receptive_field,
        norm_method=norm_method,
        n_skips=n_skips,
        n_conv_filters=32,
        n_dense_filters=128,
        input_shape=tuple(X_train.shape[1:]),
        last_only=False)


    fgbg_model_name = 'conv_fgbg_model'
    pixelwise_model_name = 'conv_edgeseg_model'

    n_epoch = args.epochs  # Number of training epochs

    lr = args.lr
    fgbg_optimizer = SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True,clipvalue=0.5)
    pixelwise_optimizer = SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True,clipvalue=0.5)

    lr_sched = rate_scheduler(lr=lr, decay=0.99)

    batch_size = 1  # fully convolutional training uses 1 image per batch

    datagen = ImageFullyConvDataGenerator(
        rotation_range=180,
        zoom_range=(.8, 1.2),
        horizontal_flip=True,
        vertical_flip=True)

    datagen_val = ImageFullyConvDataGenerator()

    # Create the foreground/background data iterators
    fgbg_train_data = datagen.flow(
        {'X': X_train, 'y': y_train},
        seed=seed,
        skip=n_skips,
        transform='fgbg',
        batch_size=batch_size)

    fgbg_val_data = datagen_val.flow(
        {'X': X_val, 'y': y_val},
        seed=seed,
        skip=n_skips,
        transform='fgbg',
        batch_size=batch_size)

    # Create the pixelwise data iterators
    pixelwise_train_data = datagen.flow(
        {'X': X_train, 'y': y_train},
        seed=seed,
        skip=n_skips,
        transform='pixelwise',
        transform_kwargs=pixelwise_kwargs,
        batch_size=batch_size)

    pixelwise_val_data = datagen_val.flow(
        {'X': X_val, 'y': y_val},
        seed=seed,
        skip=n_skips,
        transform='pixelwise',
        transform_kwargs=pixelwise_kwargs,
        batch_size=batch_size)
    
    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
                    y_true, y_pred,
                    n_classes=2,
                    from_logits=False)

    fgbg_model.compile(
        loss=loss_function,
        optimizer=fgbg_optimizer,
        metrics=['accuracy'])

    model_path = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_model_name))
    loss_path = os.path.join(MODEL_DIR, '{}.npz'.format(fgbg_model_name))
    num_gpus = count_gpus()
    print('Training on', num_gpus, 'GPUs.')
    
    train_callbacks = get_callbacks(
        model_path,
        lr_sched=lr_sched,
        save_weights_only=True,
        monitor='val_loss',
        verbose=1)

    loss_history = fgbg_model.fit(
        fgbg_train_data,
        steps_per_epoch=fgbg_train_data.y.shape[0] // batch_size,
        epochs=25,
        validation_data=fgbg_val_data,
        validation_steps=fgbg_val_data.y.shape[0] // batch_size,
        callbacks=train_callbacks)
    
    # Create the pixelwise FeatureNet Model
    pixelwise_model = model_zoo.bn_feature_net_skip_2D(
        fgbg_model=fgbg_model,
        n_features=4 if separate_edge_classes else 3,
        receptive_field=receptive_field,
        norm_method=norm_method,
        n_skips=n_skips,
        n_conv_filters=32,
        n_dense_filters=128,
        last_only=False,
        input_shape=tuple(X_train.shape[1:]))
    
    def loss_function(y_true, y_pred):
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred,
            n_classes=4 if separate_edge_classes else 3,
            from_logits=False)

    pixelwise_model.compile(
        loss=loss_function,
        optimizer=pixelwise_optimizer,
        metrics=['accuracy'])

    model_path = os.path.join(MODEL_DIR, '{}.h5'.format(pixelwise_model_name))
    loss_path = os.path.join(MODEL_DIR, '{}.npz'.format(pixelwise_model_name))

    num_gpus = count_gpus()

    print('Training on', num_gpus, 'GPUs.')
    train_callbacks = get_callbacks(
        model_path,
        lr_sched=lr_sched,
        save_weights_only=True,
        monitor='val_loss',
        verbose=1)

    loss_history = pixelwise_model.fit(
        pixelwise_train_data,
        steps_per_epoch=pixelwise_train_data.y.shape[0] // batch_size,
        epochs=100,
        validation_data=pixelwise_val_data,
        validation_steps=pixelwise_val_data.y.shape[0] // batch_size,
        callbacks=train_callbacks)

    print("finish trainning")
    
if __name__ == '__main__':
    main()