import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from PIL import Image
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import yaml
import time
import resource
import time
from data.dataset import PatientDataset
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve

start_time = time.time()

# Load configuration from file
config_path = os.path.join(".", "config_matched.yml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
print('Finish loading the config')
# Print to verify
# print(config)

# Accessing configuration parameters
image_base_path = config['data']['image_base_path']
train_csv = config['data']['train_data_path']
val_csv = config['data']['val_data_path']
test_csv = config['data']['test_data_path']
batch_size = config['data']['batch_size']
num_workers = config['data']['num_workers']
input_size = config['model']['input_size']
num_classes = config['model']['num_classes']
input_channels = config['model']['input_dimensions']['input_channels']
depth = config['model']['input_dimensions']['depth']
height = config['model']['input_dimensions']['height']
width = config['model']['input_dimensions']['width']
num_epochs = config['training_parameters']['epochs']
save_model_path = config['training_parameters']['pretrain']


# Create datasets
train_dataset = PatientDataset(valid_id_dir=train_csv, image_base_path=image_base_path, transform=None)
val_dataset = PatientDataset(valid_id_dir=val_csv, image_base_path=image_base_path, transform=None)
test_dataset = PatientDataset(valid_id_dir=test_csv, image_base_path=image_base_path, transform=None)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('Finish creating data loaders')

class ComplexCNN3D(nn.Module):
    def __init__(self):
        super(ComplexCNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool3d(3, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool3d(3, stride=2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool3d(3, stride=2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=5, padding=2)
        self.pool4 = nn.MaxPool3d(3, stride=2)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=5, padding=2)
        self.pool5 = nn.MaxPool3d(3, stride=2)

        # Increasing the number of neurons in the fully connected layers
        self.fc1 = nn.Linear(175616, 8192)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(8192, 2048)  # Increased number of neurons
        self.fc3 = nn.Linear(2048, 512)  # Increased number of neurons
        self.fc4 = nn.Linear(512, 1)  # Output layer

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

net = ComplexCNN3D()
print('Finished defining class')

class EarlyStopping:
    def __init__(self, patience=8, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = None
        self.delta = delta  # Minimum change to qualify as an improvement

    def __call__(self, val_loss, model):
        if self.val_loss_min is None:
            self.val_loss_min = val_loss
            self.best_score = -val_loss
        elif abs(val_loss - self.val_loss_min) <= self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.val_loss_min = val_loss
            self.best_score = -val_loss
            self.counter = 0

        if self.verbose and self.counter == 0:
            print(f'Validation loss decreased ({self.val_loss_min:.6f}). Resetting early stopping counter.')

def save_checkpoint(state, filename):
    filepath = os.path.join('/external/rprshnas01/netdata_kcni/dflab/team/evh/projects/checkpoints_mri', filename)
    torch.save(state, filepath)

            
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.007, momentum=0.8, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
early_stopping = EarlyStopping(patience=8, verbose=True, delta=0.0)

train_losses = []
val_losses = []

best_val_loss = float('inf')

for epoch in range(100):
    running_loss = 0.0
    net.train()
    
    for i, (images, labels) in enumerate(train_loader, 0):

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(images)
        outputs = outputs.view(-1, 1)  # Reshape outputs to [batch_size, 1]
        print('output=', outputs.detach().cpu().numpy())
        
        # Compute accuracy
        predicted = (outputs > 0.5).float()  # Convert outputs to binary predictions
        accuracy = accuracy_score(labels, predicted)
        
        print(f"Training Accuracy: {accuracy * 100:.2f}%")
        
        loss = criterion(outputs, labels)
        loss.backward()
        print('loss = ', loss.item())
        print('label = ', outputs.detach().cpu().numpy())
        
        optimizer.step()
        
        running_loss += loss.item()

    scheduler.step()

    # Average training loss for the epoch
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation loss
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels
            images = images

            outputs = net(images)
            outputs = outputs.view(-1, 1)  # Reshape outputs to [batch_size, 1]
            # print(outputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch + 1}, Training Loss: {epoch_train_loss}, Validation Loss: {val_loss}')
    early_stopping(val_loss, net)

    if early_stopping.early_stop:
        print('Early stopping')
        break

# Save the final model
save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses
}, filename='final_model_0.007_trial2.pth.tar')

print('Finished Training')

# Compute accuracy and other metrics
all_labels = []
all_outputs = []
all_predicted = []

with torch.no_grad():
    for images, labels in val_loader:
        labels = labels
        images = images

        # Forward pass
        outputs = net(images)
        outputs = outputs.view(-1, 1)  # Reshape outputs to [batch_size, 1]

        # Compute loss (optional, if you need it for reporting)
        loss = criterion(outputs, labels)

        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.view(-1)

        # Collecting all labels and outputs for metric computation
        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())
        predicted = (outputs > 0.5).float()
        all_predicted.extend(predicted.cpu().numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_outputs = np.array(all_outputs)
all_predicted = np.array(all_predicted)

# Number of bootstrap samples
n_bootstraps = 500
rng_seed = 10000  # control reproducibility
rng = np.random.default_rng(rng_seed)

# Compute metrics on the original dataset
conf_matrix = confusion_matrix(all_labels, all_predicted)
precision = precision_score(all_labels, all_predicted)
recall = recall_score(all_labels, all_predicted)
f1 = f1_score(all_labels, all_predicted)
roc_auc = roc_auc_score(all_labels, all_outputs)
accuracy = accuracy_score(all_labels, all_predicted)

# Bootstrapping for ROC-AUC and AUPRC
bootstrapped_aucs_roc = []
bootstrapped_aucs_pr = []
for i in range(n_bootstraps):
    # Bootstrap by sampling with replacement
    indices = rng.choice(len(all_labels), len(all_labels), replace=True)
    if len(np.unique(all_labels[indices])) < 2:
        # We need at least one positive and one negative sample
        continue
    # Calculate ROC-AUC
    score_roc = roc_auc_score(all_labels[indices], all_outputs[indices])
    # Calculate PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels[indices], all_outputs[indices])
    score_pr = auc(recall_vals, precision_vals)
    
    # Store the scores
    bootstrapped_aucs_roc.append(score_roc)
    bootstrapped_aucs_pr.append(score_pr)

# Convert to numpy arrays for easier computation of statistics
bootstrapped_aucs_roc = np.array(bootstrapped_aucs_roc)
bootstrapped_aucs_pr = np.array(bootstrapped_aucs_pr)

# Compute standard deviations
std_auc_roc = bootstrapped_aucs_roc.std()
std_auc_pr = bootstrapped_aucs_pr.std()

# Compute 95% confidence intervals (2.5th and 97.5th percentiles)
ci_lower_roc = np.percentile(bootstrapped_aucs_roc, 2.5)
ci_upper_roc = np.percentile(bootstrapped_aucs_roc, 97.5)
ci_lower_pr = np.percentile(bootstrapped_aucs_pr, 2.5)
ci_upper_pr = np.percentile(bootstrapped_aucs_pr, 97.5)

# Compute original ROC-AUC and PR-AUC on the whole dataset
fpr, tpr, thresholds_roc = roc_curve(all_labels, all_outputs)
roc_auc_value = auc(fpr, tpr)

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(all_labels, all_outputs)
pr_auc_value = auc(recall_vals, precision_vals)

# Printing the results
print(f"ROC-AUC: {roc_auc_value:.4f} ± {std_auc_roc:.4f}")
print(f"95% CI for ROC-AUC: [{ci_lower_roc:.4f}, {ci_upper_roc:.4f}]")
print(f"PR-AUC: {pr_auc_value:.4f} ± {std_auc_pr:.4f}")
print(f"95% CI for PR-AUC: [{ci_lower_pr:.4f}, {ci_upper_pr:.4f}]")

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

save_dir = './plots'
# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
roc_curve_path = f'{save_dir}/roc_curve_0.007_trial2.png'
plt.savefig(roc_curve_path)
plt.close()

# Plotting the Precision-Recall curve
plt.figure()
plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (area = {pr_auc_value:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
pr_curve_path = f'{save_dir}/pr_curve_0.007_trial2.png'
plt.savefig(pr_curve_path)
plt.close()

# Plotting the loss vs epochs
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()
loss_epoch_path = f'{save_dir}/loss_epoch_0.007_trial2.png'
plt.savefig(loss_epoch_path)
plt.close()

# Calculate optimal threshold using Youden's J statistic
youden_j = tpr - fpr
optimal_threshold_index = np.argmax(youden_j)
optimal_threshold = thresholds_roc[optimal_threshold_index]

print(f"Optimal Threshold: {optimal_threshold}")

# Evaluate metrics at the optimal threshold
all_predicted_optimal = (all_outputs >= optimal_threshold).astype(int)
conf_matrix_optimal = confusion_matrix(all_labels, all_predicted_optimal)
precision_optimal = precision_score(all_labels, all_predicted_optimal)
recall_optimal = recall_score(all_labels, all_predicted_optimal)
f1_optimal = f1_score(all_labels, all_predicted_optimal)
accuracy_optimal = accuracy_score(all_labels, all_predicted_optimal)

print("Metrics at Optimal Threshold:")
print(f"Confusion Matrix: \n{conf_matrix_optimal}")
print(f"Precision: {precision_optimal:.4f}")
print(f"Recall: {recall_optimal:.4f}")
print(f"F1 Score: {f1_optimal:.4f}")
print(f"Accuracy: {accuracy_optimal * 100:.2f}%")

# Plotting the ROC curve with optimal threshold
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', label='Optimal Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
roc_curve_path_optimal = f'{save_dir}/roc_curve_optimal_0.007_trial2.png'
plt.savefig(roc_curve_path_optimal)
plt.close()

# Plotting the Precision-Recall curve with optimal threshold
plt.figure()
plt.plot(recall_vals, precision_vals, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.scatter(recall_vals[optimal_threshold_index], precision_vals[optimal_threshold_index], color='red', label='Optimal Threshold')
pr_curve_path_optimal = f'{save_dir}/pr_curve_optimal_0.007_trial2.png'
plt.savefig(pr_curve_path_optimal)
plt.close()


print("--- %s seconds elapsed ---" % (time.time() - start_time))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)