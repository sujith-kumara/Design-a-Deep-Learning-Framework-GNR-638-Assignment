import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=False):
	"""
	Returns a model with final layer adjusted to num_classes.
	Defaults to ResNet18; change if you have a custom CNN.
	"""
	model = models.resnet18(pretrained=pretrained)
	# replace final fc
	in_features = model.fc.in_features
	model.fc = nn.Linear(in_features, num_classes)
	return model
