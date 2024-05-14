import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import car_data
from model import resnext50_32x4d
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = "./datasets/car/train"
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = car_data.Car(root=root, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = resnext50_32x4d(num_classes=10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in progress_bar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / (i + 1):.3f} | Acc: {100 * correct / total:.3f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(f'Epoch {epoch + 1} complete! Average Loss: {epoch_loss:.3f} | Accuracy: {epoch_acc:.3f}%')

    if epoch_loss < 0.005:
        break

torch.save(model.state_dict(), "resnext50_model.pth")
print('Finished Training')
