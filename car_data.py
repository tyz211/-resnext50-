from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import os
from PIL import Image


class Car(Dataset):
    def __init__(self, root, transforms=None):
        imgs = []
        for folder_name in os.listdir(root):
            if folder_name not in ["truck", "taxi", "minibus", "fire engine", "racing car", "SUV", "bus", "jeep",
                                   "family sedan", "heavy truck"]:
                print(f"data label error: {folder_name}")
                continue  

            label_map = {
                "truck": 0,
                "taxi": 1,
                "minibus": 2,
                "fire engine": 3,
                "racing car": 4,
                "SUV": 5,
                "bus": 6,
                "jeep": 7,
                "family sedan": 8,
                "heavy truck": 9,
            }
            label = label_map[folder_name]
            childpath = os.path.join(root, folder_name)
            for imgname in os.listdir(childpath):
                imgpath = os.path.join(childpath, imgname)
                imgs.append((imgpath, label))

        self.imgs = imgs
        self.transforms = transforms if transforms is not None else T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert("RGB")  # 确保图像为RGB格式
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    root = "./datasets/car/train"
    train_dataset = Car(root)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for data, label in train_dataset:
        print(data.shape)
        pass