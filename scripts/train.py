import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_dataloaders
from src.model import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for X, y in loader:
		X, y = X.to(device), y.to(device)
		optimizer.zero_grad()
		out = model(X)
		loss = criterion(out, y)
		loss.backward()
		optimizer.step()
		running_loss += loss.item() * X.size(0)
		_, preds = out.max(1)
		correct += (preds == y).sum().item()
		total += y.size(0)
	return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for X, y in loader:
			X, y = X.to(device), y.to(device)
			out = model(X)
			loss = criterion(out, y)
			running_loss += loss.item() * X.size(0)
			_, preds = out.max(1)
			correct += (preds == y).sum().item()
			total += y.size(0)
	return running_loss / total, correct / total

def run_for_dataset(dataset_name, args, device):
	print(f"==> Processing dataset: {dataset_name}")
	train_loader, val_loader, num_classes = get_dataloaders(dataset_name, data_root=args.data_root,
															batch_size=args.batch_size, input_size=args.input_size)
	model = get_model(num_classes, pretrained=args.pretrained).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

	best_acc = 0.0
	out_dir = os.path.join(args.save_dir, dataset_name)
	os.makedirs(out_dir, exist_ok=True)
	best_path = os.path.join(out_dir, 'best.pth')

	if args.evaluate and os.path.exists(best_path):
		model.load_state_dict(torch.load(best_path, map_location=device))
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		print(f"Eval {dataset_name} -> loss: {val_loss:.4f}, acc: {val_acc:.4f}")
		return

	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		scheduler.step()
		print(f"[{dataset_name}] Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(model.state_dict(), best_path)
	print(f"Finished {dataset_name}. Best val acc: {best_acc:.4f}. Model saved to {best_path}")

def parse_args():
	parser = argparse.ArgumentParser(description="Train/test separately on data_1 and data_2")
	parser.add_argument('--dataset', choices=['data_1', 'data_2', 'all'], default='all')
	parser.add_argument('--data-root', default='data', help='root folder containing datasets')
	parser.add_argument('--save-dir', default='outputs', help='where to write checkpoints')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--input-size', type=int, default=224)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--pretrained', action='store_true')
	parser.add_argument('--evaluate', action='store_true', help='only run evaluation (requires saved checkpoint)')
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	return parser.parse_args()

def main():
	args = parse_args()
	device = torch.device(args.device)
	targets = ['data_1', 'data_2'] if args.dataset == 'all' else [args.dataset]
	for ds in targets:
		try:
			run_for_dataset(ds, args, device)
		except FileNotFoundError as e:
			print(f"Skipping {ds} -> {e}")

if __name__ == '__main__':
	main()
