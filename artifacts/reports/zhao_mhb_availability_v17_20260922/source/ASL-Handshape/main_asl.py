import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Handshape.model.asl_handshape_gcn import GCNClassifier
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import json

class ImageDataset(Dataset):
    def __init__(self, data_file, label_file=None):
        """Load data. If label_file is provided, load labels; otherwise only load data for prediction."""
        self.data = np.load(data_file, allow_pickle=True)

        # Convert to numerical arrays
        if isinstance(self.data, np.ndarray) and self.data.dtype == np.object_:
            self.data = np.array([np.array(frame, dtype=np.float32) for frame in self.data])

        self.data = torch.tensor(self.data, dtype=torch.float32)

        # Load labels if provided
        if label_file:
            with open(label_file, 'rb') as f:
                self.labels = pickle.load(f)[1]
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

def train_ensemble(models, train_loader, criterion, optimizers, device, temperature=1.0):
    """Train each model in the ensemble with temperature-scaled cross entropy."""
    for model in models:
        model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            outputs = model(data)

            # Apply temperature scaling
            scaled_outputs = outputs / temperature
            loss = criterion(scaled_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Top-1 Accuracy
            _, predicted = scaled_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

def evaluate_ensemble(models, val_loader, criterion, device):
    """Evaluate ensemble performance on validation set, returning Top-1 and Top-5 accuracy."""
    for model in models:
        model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = sum([model(data) for model in models]) / len(models)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Top-1
            _, predicted_top1 = outputs.topk(1, dim=1)
            correct_top1 += predicted_top1.squeeze(1).eq(labels).sum().item()

            # Top-5
            _, predicted_top5 = outputs.topk(5, dim=1)
            correct_top5 += predicted_top5.eq(labels.unsqueeze(1)).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    top1_accuracy = 100.0 * correct_top1 / total
    top5_accuracy = 100.0 * correct_top5 / total

    return avg_loss, top1_accuracy, top5_accuracy

def predict_ensemble(models, test_loader, device):
    """Run prediction on test set and return top-10 results per frame."""
    for model in models:
        model.eval()

    all_top10_results = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            batch_size, seq_len, num_points, num_features = data.shape
            data = data.view(-1, num_points, num_features)

            outputs = sum([model(data) for model in models]) / len(models)
            probs = F.softmax(outputs, dim=1)

            top10_probs, top10_indices = probs.topk(10, dim=1, largest=True, sorted=True)

            top10_probs = top10_probs.view(batch_size, seq_len, 10).cpu().tolist()
            top10_indices = top10_indices.view(batch_size, seq_len, 10).cpu().tolist()

            batch_top10 = []
            for b in range(batch_size):
                seq_top10 = []
                for t in range(seq_len):
                    cls_list = top10_indices[b][t]
                    prob_list = top10_probs[b][t]
                    seq_top10.append([cls_list, prob_list])
                batch_top10.append(seq_top10)

            all_top10_results.extend(batch_top10)

    return all_top10_results

def compute_accuracy(predictions_top1, predictions_top10, labels):
    """Calculate Top-1 and Top-10 accuracy."""
    correct_top1 = np.sum(predictions_top1 == labels)
    total = labels.size
    top1_accuracy = 100.0 * correct_top1 / total

    correct_top10 = np.sum(np.any(predictions_top10 == labels[..., None], axis=2))
    top10_accuracy = 100.0 * correct_top10 / total

    return top1_accuracy, top10_accuracy

def main(mode="train"):
    """Train, validate, or test the model."""
    train_data_file = "data/2_25_train_combined_data_frames.npy"
    train_label_file = "data/2_25_train_combined_label.pkl"
    test_data_file = "data/val_handshape_dominant.npy"
    output_dir = "save"
    save_json = "json/top10_results.json"
    model_path = os.path.join(output_dir, "asl_model.pt")

    batch_size = 1024
    learning_rate = 0.0001
    num_epochs = 1000
    num_models = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = {
        "num_class": 88,
        "num_point": 11,
        "graph": "graph.sign_10.Graph",
        "groups": 16,
        "graph_args": {"labeling_mode": "spatial"},
    }

    models = [GCNClassifier(**model_args).to(device) for _ in range(num_models)]
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    criterion = torch.nn.CrossEntropyLoss()
    schedulers = [StepLR(optimizer, step_size=1000, gamma=0.1) for optimizer in optimizers]

    if os.path.exists(model_path):
        state_dicts = torch.load(model_path)
        for model, state_dict in zip(models, state_dicts):
            model.load_state_dict(state_dict)
        print(f"Model parameters loaded from {model_path}")

    if mode == "train":
        print("Starting training...")

        train_dataset = ImageDataset(train_data_file, train_label_file)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_ensemble(models, train_loader, criterion, optimizers, device, temperature=5.0)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            for scheduler in schedulers:
                scheduler.step()

        os.makedirs(output_dir, exist_ok=True)
        torch.save([model.state_dict() for model in models], model_path)
        print(f"Model parameters saved to {model_path}")

    elif mode == "test":
        print("Starting testing...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")

        test_dataset = ImageDataset(test_data_file)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions_top10 = predict_ensemble(models, test_loader, device)

        with open(save_json, 'w') as f:
            json.dump(predictions_top10, f)

    elif mode == "valid":
        print("Starting validation training...")

        full_dataset = ImageDataset(train_data_file, train_label_file)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_ensemble(models, train_loader, criterion, optimizers, device, temperature=2.0)
            val_loss, val_top1, val_top5 = evaluate_ensemble(models, val_loader, criterion, device)

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Top-1 Acc: {val_top1:.2f}%, Top-5 Acc: {val_top5:.2f}%")

            for scheduler in schedulers:
                scheduler.step()

        os.makedirs(output_dir, exist_ok=True)
        torch.save([model.state_dict() for model in models], model_path)
        print(f"Model parameters saved to {model_path}")

    else:
        raise ValueError("Invalid mode. Please choose 'train', 'test', or 'valid'.")


if __name__ == "__main__":
    # main(mode="train")
    # main(mode="valid")
    main(mode="test")
