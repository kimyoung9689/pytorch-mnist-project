import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 하이퍼파라미터 및 장치 설정
batch_size = 64
learning_rate = 0.001
epochs = 10
device = torch.device("cpu")
print(f"Using device: {device}")

# 2. CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(in_features=7*7*32, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 7*7*32)
        x = self.fc1(x)
        return x

# 3. 모델 학습 함수
def train_model(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 4. 모델 평가 함수
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 5. 예측 시각화 함수
def predict_and_visualize(model, test_loader):
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig = plt.figure(figsize=(12, 6))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        img = np.squeeze(images[i].numpy())
        ax.imshow(img, cmap='gray')

        title_color = "green" if predicted[i].item() == labels[i].item() else "red"
        ax.set_title(f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}", color=title_color)

    plt.tight_layout()
    plt.show()

# 6. 메인 실행 블록
if __name__ == "__main__":
    # 데이터 로더 준비
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 설정
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습 및 평가
    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch}/{epochs} ---")
        train_model(model, device, train_loader, optimizer, criterion)
        evaluate_model(model, device, test_loader, criterion)
    
    # 학습된 모델 저장
    torch.save(model.state_dict(), "mnist_cnn_model.pth")
    print("\nModel successfully trained and saved to mnist_cnn_model.pth")

    # 저장된 모델 불러와서 예측 시각화
    print("\n--- Saved model loaded for visualization ---")
    loaded_model = CNN()
    loaded_model.load_state_dict(torch.load("mnist_cnn_model.pth"))
    loaded_model.eval()
    predict_and_visualize(loaded_model, test_loader)
