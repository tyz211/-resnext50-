import torch
from torchvision import models, transforms
import os
from PIL import Image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.resnext50_32x4d(pretrained=False)

num_classes = 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)


model.load_state_dict(torch.load('resnext50_model2.pth', map_location=device))


model = model.to(device)

model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def calculate_accuracy(path):
    total_images = 0
    correct_predictions = 0
    dict = {0: 'truck', 1: 'taxi', 2: 'minibus', 3: 'fire engine',
            4: 'racing car', 5: 'SUV', 6: 'bus', 7: 'jeep', 8: 'family sedan', 9: 'heavy truck'}

    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                prediction = predict(image_path)
                if dict[prediction] == class_name:
                    correct_predictions += 1
                total_images += 1

    accuracy = correct_predictions / total_images
    return accuracy


if __name__ == "__main__":
    accuracy = calculate_accuracy('./datasets/car/val')
    print("Accuracy:", accuracy)
