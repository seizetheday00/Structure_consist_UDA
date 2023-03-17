from ast import Num
from pydoc import classname
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import math

from common.modules.classifier import Classifier as ClassifierBase
from ..modules.entropy import entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['StructurePreserveLoss', 'ImageClassifier']


class StructurePreserveLoss(nn.Module):
    def __init__(self, num_classes: Optional[int] = -1,
    features_dim: Optional[int] = -1):
        super(StructurePreserveLoss, self).__init__()
        self.num_classes = num_classes
        self.features_fim = features_dim

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, \
        g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:

        batch_size, num_classes = g_s.shape
        batch_size, features_dim = f_s.shape
        g_s_label = F.softmax(g_s,dim=1)
        g_t_label = F.softmax(g_t,dim=1)

        g_s_label = torch.argmax(g_s_label, -1)
        g_t_label = torch.argmax(g_t_label, -1)

        classmeans = torch.zeros((num_classes, f_s.size(1)))
        for i in range(num_classes):
            if (len(f_s[g_s_label==i,:])==0):
                continue
            else:
                classmeans[i,:] = torch.mean(f_s[g_s_label==i,:],dim=0,keepdim=True)
        classmeans = classmeans.to(device)


        distclassmeans = torch.cdist(f_t, classmeans, p=2)

        cluster = KMeans(n_clusters=num_classes, random_state=0)
        f_t_cpu = f_t.detach().cpu().numpy()
        cluster = cluster.fit(f_t_cpu)

        targetclassmeans = cluster.cluster_centers_
        targetclassmeans = torch.tensor(targetclassmeans).to(device)

        distclustermeans = torch.cdist(f_t.to(torch.float32),targetclassmeans.to(torch.float32))
        expmatclass = torch.exp(-distclassmeans)
        expmatcluster = torch.exp(-distclustermeans)

        probmatfinal = torch.max(expmatclass, expmatcluster)

        spl_acc = F.cross_entropy(probmatfinal, g_t_label)
        return spl_acc



class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
