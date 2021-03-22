import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models import experimental

if __name__ == '__main__':
    models = experimental.attempt_load("weights/yolov5s.pt")
    convs = [(name, module) for (name, module) in models.named_modules() if isinstance(module, nn.Conv2d)]
    for (name, conv) in convs:
        sns.displot(conv.weight.reshape(-1), rug=True, kind="kde")
        plt.savefig("plots/convs/{}.png".format(name))
        plt.close('all')
        sns.displot(conv.bias.reshape(-1), rug=True, kind="kde")
        plt.savefig("plots/bias/{}.png".format(name))
        plt.close('all')
