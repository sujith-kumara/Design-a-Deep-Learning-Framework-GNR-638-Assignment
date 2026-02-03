import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name, data_root='data', batch_size=64, input_size=224, num_workers=4):
	"""
	Expect layout: {data_root}/{dataset_name}/train/* and .../test/*
	returns: train_loader, val_loader, num_classes
	"""
	dataset_dir = os.path.join(data_root, dataset_name)
	train_dir = os.path.join(dataset_dir, 'train')
	val_dir = os.path.join(dataset_dir, 'test')

	# ...validate directories...
	if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
		raise FileNotFoundError(f"Expected train/ and test/ under {dataset_dir}")

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	train_tf = transforms.Compose([
		transforms.RandomResizedCrop(input_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])
	val_tf = transforms.Compose([
		transforms.Resize(int(input_size * 1.15)),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		normalize,
	])

	train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
	val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

	num_classes = len(train_ds.classes)
	return train_loader, val_loader, num_classes
