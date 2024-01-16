import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
import argparse
import numpy as np

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Stardist Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default='path/to/your/dataset/directory', 
                        help='path to the data')
    parser.add_argument('--exp-name', type=str, default='stardist_train_on_all_types_wholecell_1C', 
                        help='name of the experiment')
    parser.add_argument('--model-name', type=str, default='stardist', 
                        help='name of the experiment')
    parser.add_argument('--log-dir', type=str, default='./stardist_logs', 
                        help='where you want to put your logs')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Start training with existing weights')
    parser.add_argument('--chooseone', default='All',
                        help='choose from Tonsil Breast BV2 Epidermis SHSY5Y Esophagus \
                        A172 SkBr3 Lymph_Node Lung SKOV3 Pancreas Colon MCF7 Huh7 BT474 lymph node metastasis Spleen')                           
    args = parser.parse_args()
    # from deepcell_toolbox.multiplex_utils import multiplex_preprocess

    # create folder for this set of experiments
    experiment_folder = args.exp_name
    MODEL_DIR = os.path.join("./Stardist", experiment_folder)

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    np.random.seed(42)
    lbl_cmap = random_label_cmap()
    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    # use_gpu = False and gputools_available()
    use_gpu = False and gputools_available()

    EXP_NAME = args.exp_name
    
    print("Stardist")
    print("loading data")
    data_dir="path/to/your/dataset/directory"
    
    val_dict = np.load(os.path.join(data_dir, 'tissuenet_val_split_256x256_memserpreprocess.npz'), allow_pickle=True)
    train_dict = np.load(os.path.join(data_dir, 'tissuenet_train_split_256x256_memserpreprocess.npz'), allow_pickle=True)

    X_train = train_dict['X'][...,1].squeeze()
    y_train = train_dict['y'][...,0].squeeze()

    X_val = val_dict['X'][...,1].squeeze()
    y_val = val_dict['y'][...,0].squeeze()

    tissue_type_train = train_dict['tissue_list']
    tissue_type_val = val_dict['tissue_list']
    y_train = y_train.astype('int32').squeeze()
    y_val = y_val.astype('int32').squeeze()

    axis_norm = (0,1) 
    n_channel = 1 if X_train[0].ndim == 2 else X_train[0].shape[-1]

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X_train = [normalize(x, 0, 99.8, axis=axis_norm) for x in tqdm(X_train)]
    y_train = [fill_label_holes(y) for y in tqdm(y_train)]

    X_val = [normalize(x, 0, 99.8, axis=axis_norm) for x in tqdm(X_val)]
    y_val = [fill_label_holes(y) for y in tqdm(y_val)]

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2, 2)

    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = 1,
    )
    # print(conf)
    # vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8, total_memory=40536)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)
    
    model = StarDist2D(conf, name=EXP_NAME, basedir=MODEL_DIR)
    # print(model)
    median_size = calculate_extents(list(y_train), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    def random_fliprot(img, mask): 
        assert img.ndim >= mask.ndim
        axes = tuple(range(mask.ndim))
        perm = tuple(np.random.permutation(axes))
        img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
        mask = mask.transpose(perm) 
        for ax in axes: 
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask 

    def random_intensity_change(img):
        img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        return img

    def augmenter(x, y):
        """Augmentation of a single input/label image pair.
        x is an input image
        y is the corresponding ground-truth label image
        """
        x, y = random_fliprot(x, y)
        x = random_intensity_change(x)
        # add some gaussian noise
        sig = 0.02*np.random.uniform(0,1)
        x = x + sig*np.random.normal(0,1,x.shape)
        return x, y

    model.train(X_train, y_train, validation_data=(X_val, y_val), augmenter=augmenter, epochs=args.epochs)
    print("finish trainning")

if __name__ == '__main__':
    main()











