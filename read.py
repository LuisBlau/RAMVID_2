import numpy as np
import torch.utils.data as data
import torch
data = np.load('mnist_test_seq.npy')
split = 1000
training_set = torch.from_numpy(
            np.load('mnist_test_seq.npy').swapaxes(0, 1)
        )
test_set = torch.from_numpy(
            np.load('mnist_test_seq.npy').swapaxes(0, 1)[:split]
        )
print(test_set[0].shape)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=training_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

