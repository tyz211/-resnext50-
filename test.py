import torch
from torchvision import models, transforms
from tkinter import filedialog, Tk, Label, Button
from PIL import Image
import tkinter as tk


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.resnext50_32x4d(pretrained=False)

num_classes = 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)


model.load_state_dict(torch.load('resnext50_model.pth', map_location=device))


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


def select_image():
    path = filedialog.askopenfilename()
    if path:
        prediction = predict(path)
        dict = {0: 'truck', 1: 'taxi', 2: 'minibus', 3: 'fire engine',
                4: 'racing car', 5: 'SUV', 6: 'bus', 7: 'jeep', 8: 'family sedan', 9: 'heavy truck'}
        label_predict.config(text=f'Predicted class: {dict[prediction]}')

root = Tk()
root.title('ResNeXt Vehicle Classification')


frame = tk.Frame(root)
frame.pack()


label_predict = Label(frame, text='Select an image', padx=10, pady=10)
label_predict.pack()


button = Button(frame, text='Select image', command=select_image)
button.pack()

root.mainloop()
