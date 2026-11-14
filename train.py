"""train.py - Train a small CNN on the MNIST *train* split and report clean accuracy on the *test* split."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

SEED = 0
EPOCHS = 5
BATCH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    """Small convolutional network. Input pixels are in [0, 1]."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop(torch.flatten(x, 1))
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits


def main():
    torch.manual_seed(SEED)
    tf = transforms.ToTensor()  # scales pixels to [0, 1]
    train_set = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = Net().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"Epoch {epoch}: test accuracy = {correct / len(test_set):.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Saved model to mnist_cnn.pt")


if __name__ == "__main__":
    main()
