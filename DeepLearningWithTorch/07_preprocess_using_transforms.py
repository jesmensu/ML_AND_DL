# torchvision.transforms provides image transformation functions that help preprocess images 
# before feeding them into deep learning models.

import torchvision.transforms as transforms
from PIL import Image


# ======= Convert image to PyTorch tensor ===========
transform = transforms.ToTensor()  
image = Image.open("image.jpg")  # Open an image
tensor_image = transform(image)

print(tensor_image.shape)  # Output: torch.Size([C, H, W])


# ============ Normalize images ============
transform = transforms.Normalize(mean=[0.5], std=[0.5])  # Standardizes pixel values to [-1, 1]
normalized_image = transform(tensor_image)

transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # For RGB images


# =========== Resize and Crop ================
transform = transforms.Resize((128, 128))  # Resize to 128x128
resized_image = transform(image)
transform = transforms.CenterCrop(100)  # Crop center 100x100 pixels
cropped_image = transform(image)
transform = transforms.RandomCrop(100)  # Random crop


# =========== Data Augmentation ==============
transform = transforms.RandomHorizontalFlip(p=0.5)
transform = transforms.RandomRotation(degrees=30)
transform = transforms.ColorJitter(brightness=0.5, contrast=0.5) # Randomly alters brightness and contrast and sasuration


#  ============ Compose Multiple Transforms ============
# chain multiple transformations together using transforms.Compose
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image = Image.open("image.jpg")
transformed_image = transform(image)


# =========== Using Transforms in a DataLoader =============
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# =============== Custom Transformations ================
class CustomTransform:
    def __call__(self, img):
        return img.rotate(45)  # Rotate all images by 45 degrees

transform = transforms.Compose([
    CustomTransform(),
    transforms.ToTensor()
])







