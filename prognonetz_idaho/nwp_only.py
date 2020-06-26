# implementation of torch dataset using only NWP data and no preprocessing
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import os
import numpy as np


class PrognonetzDataSet(data.Dataset):

    # metadata
    url = "https://www.dropbox.com/s/qwjtwfzl4ei2i05/prognonetz_dataset.tar.gz?dl=1"
    filename = "prognonetz_dataset.tar.gz"
    tgz_md5 = 'b92b75a20e9e6d9890caa48defae6868'
    train_list = [
        ['Prognonetz_INL_train.npz', '93f31bc5b8b4050dc6605c1a5ad84443'],
    ]
    test_list = [
        ['Prognonetz_INL_test.npz', 'd346bd33ff3fdd84b6892903767d5ed4'],
    ]

    def __init__(self, root, split='train', download=False, small=True, input_shape=None, output_shape=None, zerofilling=True):

        # change metadata to small set
        if small:
            self.url = "https://www.dropbox.com/s/gk72osjlcjg14fh/prognonetz_dataset_small.tar.gz?dl=1"
            self.filename = "prognonetz_dataset_small.tar.gz"
            self.tgz_md5 = 'c4d4e5e91e6c9718a762bac8b496e44e'
            self.train_list = [
                ['Prognonetz_INL_train_small.npz', '1dab55823d26488165d6eddcc1d25cf0'],
            ]
            self.test_list = [
                ['Prognonetz_INL_test_small.npz', '44456c97682c3211a3ee2038368328ed'],
            ]

        # fallback to all available features
        if input_shape is None:
            self.input_shape = ['TMP', 'WSPD', 'WDIR']
        else:
            self.input_shape = input_shape

        # fallback to all available labels
        if output_shape is None:
            self.output_shape = ['RWM_690', '690_RWM', '690_CIT', 'CIT_690', 'CIT_ROV', 'ROV_CIT']
        else:
            self.output_shape = output_shape

        self.root = os.path.expanduser(root)
        self.split = split
        self.features = None
        self.labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        # specify correct file
        if self.split is 'train':
            if small:
                filename = 'Prognonetz_INL_train_small.npz'
            else:
                filename = 'Prognonetz_INL_train.npz'
        elif self.split is 'test':
            if small:
                filename = 'Prognonetz_INL_test_small.npz'
            else:
                filename = 'Prognonetz_INL_test.npz'
        # load features
        ft_index = list(np.load(os.path.join(self.root, filename))['ft_nwp_index'])
        ft_shape = [ft_index.index(item) for item in self.input_shape]
        self.features = np.load(os.path.join(self.root, filename))['ft_nwp'][..., ft_shape]

        # load labels
        lb_index = list(np.load(os.path.join(self.root, filename))['lb_index'])
        lb_shape = [lb_index.index(item) for item in self.output_shape]
        self.labels = np.load(os.path.join(self.root, filename))['lb'][:, :, lb_shape]
        
    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            # print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

