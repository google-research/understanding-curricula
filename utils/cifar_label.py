# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision.datasets import CIFAR100
import numpy as np

class CIFAR100N(CIFAR100):
    """
    Extends CIFAR100 dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 rand_fraction=0.0):
        super(CIFAR100N, self).__init__(root=root,
                                              train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)

        assert (rand_fraction <= 1.0) and (rand_fraction >= 0.0)
        self.rand_fraction = rand_fraction

        if self.rand_fraction > 0.0:
            self.targets = self.corrupt_fraction_of_data()
            
    def corrupt_fraction_of_data(self):
        """Corrupts fraction of train data by permuting image-label pairs."""

        # Check if we are not corrupting test data
        assert self.train is True, 'We should not corrupt test data.'
        rearrange = []
        length = len(self.targets)//4
        start_next = 0
        new_labels = []
        for i in range(4):
            nr_corrupt_instances = start_next + int(np.floor(length * self.rand_fraction))
            print('Randomizing {} fraction of data == {} / {}'.format(self.rand_fraction,
                                                                      nr_corrupt_instances-start_next ,
                                                                      length))
            # We will corrupt the top fraction data points
            corrupt_label = self.targets[start_next:nr_corrupt_instances]
            clean_label = self.targets[nr_corrupt_instances:start_next + length]

            # Corrupting data
            np.random.seed(111)
            rand_idx = np.random.permutation(np.arange(start_next,nr_corrupt_instances))
            corrupt_label = np.array(corrupt_label)[rand_idx-start_next]
            # Adding corrupt and clean data back together
            new_labels.extend(corrupt_label)
            new_labels.extend(clean_label)
            start_next += length
            
        return np.array(new_labels)    
    
    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        #return img, target, index
        return img, target



