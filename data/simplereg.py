import numpy as np 
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
data_path = os.path.join(root_path, 'data')

__all__ = ['SimpleRegData']

class SimpleRegData:
    def __init__(self, dataset='sunspots', train_test_split=0.5):
        np.random.seed(42)
        data = np.genfromtxt(os.path.join(data_path, dataset+'.csv'), delimiter=',', dtype=np.float64)                                                                                               
        self.X = data[:, 0][:, np.newaxis]
        self.y = data[:, 1].squeeze()
        self.N = self.X.shape[0]
        self.tr_idx = np.sort(np.random.choice(self.N, int(self.N*train_test_split), replace=False))

    def _samples(self, idx='train'):
        mask = np.zeros(self.X.size, dtype=bool)
        mask[self.tr_idx] = True
        if idx == 'test':
            mask = ~mask
        X = self.X[mask, :]
        y = self.y[mask]
        return X, y

    def train_samples(self):
        return self._samples('train')
    
    def test_samples(self):
        return self._samples('test')
