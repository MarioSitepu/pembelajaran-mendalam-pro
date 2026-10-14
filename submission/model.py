from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 224
CLASSES = ['botol_plastik', 'gelas_plastik', 'kertas_kardus', 'saset_kemasan', 'styrofoam']

def get_model():
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 5)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(round(IMAGE_SIZE * 236 / 224), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
