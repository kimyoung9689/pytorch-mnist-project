import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 하이퍼파라미터 설정 (모델 학습에 필요한 설정값들)
batch_size = 64
learning_rate = 0.001
epochs = 10

# 3. 데이터 로더 준비
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 여기에 모델 구조를 정의해야 함

    def forward(self, x):
        # 데이터가 모델을 통과하는 과정을 정의해야 함
        return x

# 5. 모델, 손실 함수, 옵티마이저 설정
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 학습 루프 구현
def train(model, device, train_loader, optimizer, criterion):
    # 학습 과정 코드를 여기에 작성해야 함
    pass

# 7. 평가 함수 구현
def evaluate(model, device, test_loader):
    # 평가 과정 코드를 여기에 작성해야 함
    pass

# 8. 메인 함수
if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch}/{epochs} ---")
        train(model, device, train_loader, optimizer, criterion)
        evaluate(model, device, test_loader)