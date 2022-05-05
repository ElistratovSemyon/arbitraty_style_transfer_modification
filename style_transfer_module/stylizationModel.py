import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from os.path import join, dirname

from .stylePredictor import StylePredictor
from .ghiasi import Ghiasi

class StylizationModel(nn.Module):
	def __init__(self, dropout=None, identity_connections=False):
		super(StylizationModel,self).__init__()
		self.transformer = Ghiasi(identity_connections)
		self.embedding = StylePredictor(dropout)
		

	def forward(self, X, style, alpha=0.5):
		# get embedding representation for style
		P_s = self.embedding(style)
		# get embedding representation for content
		P_c = self.embedding(X)
		# interpolation
		embedding = alpha * P_s + (1-alpha) * P_c
		# use model
		y = self.transformer(X, embedding, alpha)
		return y
