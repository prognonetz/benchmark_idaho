# implementation of torch dataset as used in our work.
# 
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import os
import numpy as np
from prognonetz_idaho import processing


def parse_shape(shape):
    # parses a given input shape
    out_shape = []
    vectors = []
    for item in shape:
        if type(item) is processing.Vector:
            out_shape.extend([item.magnitude, item.direction])
            vectors.append((len(out_shape)-2, len(out_shape)-1))
        else:
            out_shape.append(item)
    return out_shape, vectors


def polar_to_cartesian(inp):
    # conversion from polar to cartesian. used to convert wind speed and direction into x/y-components
    rho, phi = inp[..., 0], inp[..., 1]
    x = rho * np.cos(np.deg2rad(phi))
    y = rho * np.sin(np.deg2rad(phi))
    return np.stack([x, y], axis=-1)


def split_with_seed(length, split, seed, val_size, splittype='random', chunklength=200):
    # generates indices for randomized reproducable train-val splits
    if split is 'test':
        test_indices = [i for i in range(length)]
        return test_indices
    
    if splittype is 'random':
        sample = np.random.mtrand.RandomState(seed).choice(range(length), int(val_size*length))
        if split is 'val':
            return sample
        else:
            train_indices = [i for i in range(length) if i not in sample]
            return train_indices
    elif splittype is 'chunks':
        n_chunks = int(int(val_size*length)/chunklength)

        sample = []
        distance = int(length/(n_chunks+1))
        for chunk in range(n_chunks):
            sample.extend([chunk*distance + i for i in range(chunklength)])

        if split is 'val':
            return sample
        else:
            train_indices = [i for i in range(length) if i not in sample]
            return train_indices


class PrognonetzDataSet(data.Dataset):

    url = "https://www.dropbox.com/s/qwjtwfzl4ei2i05/prognonetz_dataset.tar.gz?dl=1"
    filename = "prognonetz_dataset.tar.gz"
    tgz_md5 = 'b92b75a20e9e6d9890caa48defae6868'
    train_list = [
        ['Prognonetz_INL_train.npz', '93f31bc5b8b4050dc6605c1a5ad84443'],
    ]
    test_list = [
        ['Prognonetz_INL_test.npz', 'd346bd33ff3fdd84b6892903767d5ed4'],
    ]

    def __init__(self, root, split='train', download=False, small=True, nwp=True, obs=True, input_shape_obs=None, 
    input_shape_nwp=None, output_shape=None, output_transform=None, input_transform_obs=None, input_transform_nwp=None, 
    split_seed=0, val_size=0.2, zerofilling=True, mask=False, age_matrix=False, nearestneighbour=None, splittype='chunks'):

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

        if output_transform is None:
            self.output_transform = processing.Identity()
        else:
            self.output_transform = output_transform

        if input_transform_obs is None:
            self.input_transform_obs = processing.Identity()
        else:
            self.input_transform_obs = input_transform_obs

        if input_transform_nwp is None:
            self.input_transform_nwp = processing.Identity()
        else:
            self.input_transform_nwp = input_transform_nwp

        if input_shape_obs is None:
            self.input_shape_obs = ['RWM_temperature_15m', 'RWM_wind_speed', 'RWM_wind_direction',
                                    'RWM_solar_radiation', 'RWM_pressure', 'RWM_humidity', 'RWM_pressure_gradient_c1',
                                    'RWM_pressure_gradient_c2', '690_temperature_15m', '690_wind_speed',
                                    '690_wind_direction', '690_solar_radiation', '690_pressure', '690_humidity',
                                    '690_pressure_gradient_c1', '690_pressure_gradient_c2', 'CIT_temperature_15m',
                                    'CIT_wind_speed', 'CIT_wind_direction', 'CIT_solar_radiation', 'CIT_pressure',
                                    'CIT_humidity', 'CIT_pressure_gradient_c1', 'CIT_pressure_gradient_c2',
                                    'ROV_temperature_15m', 'ROV_wind_speed', 'ROV_wind_direction',
                                    'ROV_solar_radiation', 'ROV_pressure', 'ROV_humidity', 'ROV_pressure_gradient_c1',
                                    'ROV_pressure_gradient_c2']
        else:
            self.input_shape_obs = input_shape_obs
        if input_shape_nwp is None:
            self.input_shape_nwp = ['TMP', 'WSPD', 'WDIR']
        else:
            self.input_shape_nwp = input_shape_nwp
        if output_shape is None:
            self.output_shape = ['RWM_690', '690_RWM', '690_CIT', 'CIT_690', 'CIT_ROV', 'ROV_CIT']
        else:
            self.output_shape = output_shape
        self.root = os.path.expanduser(root)
        self.split = split
        self.features_nwp = None
        self.features_obs = None
        self.labels = None

        if not nwp and not obs:
            raise RuntimeError('nwp and obs cannot both be false')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # specify correct file
        filename = ''
        selected_samples = None
        if self.split is 'train' or self.split is 'val':
            if small:
                filename = 'Prognonetz_INL_train_small.npz'
            else:
                filename = 'Prognonetz_INL_train.npz'
            selected_samples = []
        elif self.split is 'test':
            if small:
                filename = 'Prognonetz_INL_test_small.npz'
            else:
                filename = 'Prognonetz_INL_test.npz'

        # load data from zip files
        if nwp:
            ft_nwp_index = list(np.load(os.path.join(self.root, filename))['ft_nwp_index'])
            shape, vectors = parse_shape(self.input_shape_nwp)
            ft_nwp_shape = [ft_nwp_index.index(item) for item in shape]
            if nearestneighbour is not None:
                self.features_nwp = np.load(os.path.join(self.root, filename))['ft_nwp'][..., nearestneighbour[0], nearestneighbour[1], ft_nwp_shape]
            else:
                self.features_nwp = np.load(os.path.join(self.root, filename))['ft_nwp'][..., ft_nwp_shape]
            for vector in vectors:
                # vectorize selected values
                self.features_nwp[..., vector] = polar_to_cartesian(self.features_nwp[..., vector])
            # perform processing
            self.features_nwp = self.input_transform_nwp.process(self.features_nwp)
            pre_mask_shape = self.features_nwp.shape
            pre_mask_nans = np.isnan(self.features_nwp)
            if mask:
                # create mask of ones if not nan and zeros else
                mask_nwp = np.ones(pre_mask_shape)
                mask_nwp[pre_mask_nans] = float(0)
                # append mask to data
                self.features_nwp = np.concatenate((self.features_nwp, mask_nwp), axis=-1)
            if age_matrix:
                # load weights for weight matrix
                self.nwp_weights = np.load(os.path.join(self.root, filename))['nwp_weights']
                # create mask of ones if not nan and zeros else
                age_matrix_nwp = np.ones(pre_mask_shape)
                if nearestneighbour is not None:
                    age_matrix_nwp = age_matrix_nwp * self.nwp_weights[:, :, None]
                else:
                    age_matrix_nwp = age_matrix_nwp * self.nwp_weights[:, :, None, None, None]
                age_matrix_nwp[pre_mask_nans] = float(0)
                # append mask to data
                self.features_nwp = np.concatenate((self.features_nwp, age_matrix_nwp), axis=-1)
            if zerofilling:
                # fill nan with zero
                self.features_nwp[np.isnan(self.features_nwp)] = float(0)
        if obs:
            ft_obs_index = list(np.load(os.path.join(self.root, filename))['ft_obs_index'])
            shape, vectors = parse_shape(self.input_shape_obs)
            ft_obs_shape = [ft_obs_index.index(item) for item in shape]
            self.features_obs = np.load(os.path.join(self.root, filename))['ft_obs'][..., ft_obs_shape]
            for vector in vectors:
                self.features_obs[..., vector] = polar_to_cartesian(self.features_obs[..., vector])
            self.features_obs = self.input_transform_obs.process(self.features_obs)
            if zerofilling:
                self.features_obs[np.isnan(self.features_obs)] = float(0)

        # load labels
        lb_index = list(np.load(os.path.join(self.root, filename))['lb_index'])
        lb_shape = [lb_index.index(item) for item in self.output_shape]
        self.labels = self.output_transform.process(np.load(os.path.join(self.root, filename))['lb'][:, :, lb_shape])
        if selected_samples is not None:
            selected_samples = split_with_seed(len(self.labels), self.split, split_seed, val_size, splittype=splittype)
            self.labels = self.labels[selected_samples]
            if obs:
                self.features_obs = self.features_obs[selected_samples]
            if nwp:
                self.features_nwp = self.features_nwp[selected_samples]

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
        if self.features_nwp is None:
            return self.features_obs[index], self.labels[index]
        if self.features_obs is None:
            return self.features_nwp[index], self.labels[index]
        return [self.features_nwp[index], self.features_obs[index]], self.labels[index]
