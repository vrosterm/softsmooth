import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Data Augmentation for Training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Randomly crop images
    transforms.RandomHorizontalFlip(),     # Randomly flip images horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Normalization for Testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 data
CIFAR10_train = datasets.CIFAR10("C:/Users/walke/data", train=True, download=True, transform=transform_train)
CIFAR10_test = datasets.CIFAR10("C:/Users/walke/data", train=False, download=True, transform=transform_test)
train_loader = DataLoader(CIFAR10_train, batch_size=128, shuffle=True)  # Increased batch size
test_loader = DataLoader(CIFAR10_test, batch_size=128, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Define the model architecture (DLA)
class BasicBlock(nn.Module):
    """Basic block for DLA with residual connections and three layers."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # New conv3
        self.bn3 = nn.BatchNorm2d(out_channels)  # New bn3
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # Apply the third layer
        out += self.shortcut(x)  # Residual connection
        return self.relu(out)
    
class DLA(nn.Module):
    """Deep Layer Aggregation (DLA) model."""
    def __init__(self, num_classes=10):
        super(DLA, self).__init__()
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.level0 = self._make_level(64, 64, 1, stride=1)
        self.level1 = self._make_level(64, 128, 2, stride=2)
        self.level2 = self._make_level(128, 256, 2, stride=2)
        self.level3 = self._make_level(256, 512, 2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_level(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model_cnn = DLA(num_classes=10).to(device)

# Training and testing functions
def epoch(loader, model, opt=None, desc="Batch Progress"):
    total_loss, total_err = 0., 0.
    with tqdm(loader, desc=desc, leave=True) as batch_bar:  # Keep progress bar after completion
        for X, y in batch_bar:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
            batch_bar.set_postfix(loss=loss.item())  # Update progress bar with loss
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Initialize optimizer and learning rate scheduler
opt = optim.SGD(model_cnn.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)  # SGD with momentum
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)  # Dynamic learning rate scheduler

# Train the model
num_epochs = 30
for epoch_num in range(num_epochs):
    print(f"Epoch {epoch_num+1}/{num_epochs}")
    
    # Training loop
    model_cnn.train()
    train_err, train_loss = epoch(train_loader, model_cnn, opt, desc="Train")  # Renamed progress bar to "Train"
    
    # Testing loop
    model_cnn.eval()
    test_err, test_loss = epoch(test_loader, model_cnn, desc="Test")  # Renamed progress bar to "Test"
    
    # Step the scheduler
    scheduler.step()
    
    # Print epoch results
    print(f"Epoch {epoch_num+1}: Train Accuracy: {100 - 100*train_err:.2f}%, Test Accuracy: {100 - 100*test_err:.2f}%\n")

# Save the trained model
torch.save(model_cnn.state_dict(), "model_CIFAR10.pt")
print("Model saved.")