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

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'fc1000', 'fc500',
]


import torch.nn as nn
import torch.nn.functional as F

class FcNet(nn.Module):
    def __init__(self,hidden=1000):
        super(FcNet, self).__init__()
        self.hidden = hidden
        self.conv1 = nn.Conv2d(3, 20, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20* 16 * 16, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 20 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fc500():
    model = FcNet(hidden=500)
    return model
def fc1000():
    model = FcNet(hidden=1000)
    return model

