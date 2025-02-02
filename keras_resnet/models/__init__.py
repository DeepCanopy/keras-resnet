# -*- coding: utf-8 -*-

"""
keras_resnet.models
~~~~~~~~~~~~~~~~~~~

This module implements popular residual models.
"""

from ._1d import (
    ResNet1D,
    ResNet1D18,
    ResNet1D34,
    ResNet1D50,
    ResNet1D101,
    ResNet1D152,
    ResNet1D200
)

from ._2d import (
    ResNet2D,
    ResNet2D10,
    ResNet2D18,
    ResNet2D34,
    ResNet2D50,
    ResNet2D101,
    ResNet2D152,
    ResNet2D200
)

from ._2dxnor import (
    ResNet2Dxnor,
    ResNet2Dxnor18,
    ResNet2Dxnor34,
    ResNet2Dxnor50,
    ResNet2Dxnor101,
    ResNet2Dxnor152,
    ResNet2Dxnor200
)

from ._3d import (
    ResNet3D,
    ResNet3D18,
    ResNet3D34,
    ResNet3D50,
    ResNet3D101,
    ResNet3D152,
    ResNet3D200
)

from ._feature_pyramid_2d import (
    FPN2D,
    FPN2D18,
    FPN2D34,
    FPN2D50,
    FPN2D101,
    FPN2D152,
    FPN2D200
)

from ._time_distributed_2d import (
    TimeDistributedResNet,
    TimeDistributedResNet18,
    TimeDistributedResNet34,
    TimeDistributedResNet50,
    TimeDistributedResNet101,
    TimeDistributedResNet152,
    TimeDistributedResNet200
)

# for backwards compatibility reasons
ResNet = ResNet2D
ResNet10 = ResNet2D10
ResNet18 = ResNet2D18
ResNet34 = ResNet2D34
ResNet50 = ResNet2D50
ResNet50xnor = ResNet2Dxnor50
ResNet101 = ResNet2D101
ResNet101xnor = ResNet2Dxnor101
ResNet152 = ResNet2D152
ResNet152xnor = ResNet2Dxnor152
ResNet200 = ResNet2D200
ResNet200xnor = ResNet2Dxnor200
