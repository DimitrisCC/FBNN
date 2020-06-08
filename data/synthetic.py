import numpy as np 
from mat4py import loadmat
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
data_path = os.path.join(root_path, 'data', 'datasets.mat')

__all__ = ['SyntheticData']

class SyntheticData:
    def __init__(self, dataset='T', return_param_values=False, train_test_split=0.5):
        np.random.seed(42)
        mat = loadmat(data_path)
        ds = mat[dataset]
        self.X = np.array(ds['x'])
        self.y = np.squeeze(np.array(ds['y']))
        self.N = self.X.shape[0]
        self.tr_idx = np.sort(np.random.choice(self.N, int(self.N*train_test_split), replace=False))
        self.ret_params = return_param_values
        if return_param_values:
            params = dict()
            for k, v in ds.items():
                if k != 'x' and k != 'y':
                    params[k] = np.squeeze(np.array(v))
            self.params = params

    def _samples(self, idx='train'):
        mask = np.zeros(self.X.size, dtype=bool)
        mask[self.tr_idx] = True
        if idx == 'test':
            mask = ~mask
        X = self.X[mask, :]
        y = self.y[mask]
        if self.ret_params:
            params = dict()
            for k, v in self.params.items():
                params[k] = v[mask]
            return X, y, params
        else:
            return X, y

    def train_samples(self):
        return self._samples('train')
    
    def test_samples(self):
        return self._samples('test')
