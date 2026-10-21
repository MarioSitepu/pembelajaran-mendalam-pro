"""
submission/model.py
Kontrak Resmi Studio Proyek 2 (IF25-40401 Deep Learning ITERA)
Arsitektur V5.1 (Opsi B): Quad-Paradigm Hybrid Super-Ensemble (@ 224x224)
- Backbone A (Modern 7x7 Depthwise CNN) : torchvision.models.convnext_tiny (~53.1 MB FP16)
- Backbone B (Global Self-Attention ViT): timm deit3_small_patch16_224.fb_in22k_ft_in1k (~41.4 MB FP16)
- Backbone C (Shifted-Window ViT)       : torchvision.models.swin_s (~94.6 MB FP16)
- Backbone D (Fused-MBConv CNN)         : timm tf_efficientnetv2_s.in21k_ft_in1k (~38.5 MB FP16)
- ExifTranspose otomatis di dalam get_transform() untuk memperbaiki rotasi foto smartphone
- Built-in 6-View Multi-Scale & Aspect-Rectified TTA (Orig, Flip, Zoom88, Zoom88-Flip, Aspect-Tall, Texture-Zoom76)
- Registered buffer untuk Ensemble Weights, Temperature Scaling per Model, TTA View Weights & Logit Adjustment
- Total ukuran submission/model.pth (FP16) ~227.2 MB (sangat aman di bawah batas 500 MB evaluate.py)
"""

from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm

MODEL_A_NAME = "convnext_tiny"
MODEL_B_NAME = "deit3_small_patch16_224.fb_in22k_ft_in1k"
MODEL_C_NAME = "swin_s"
MODEL_D_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
NUM_CLASSES = 5
IMG_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ExifTranspose:
    """
    Memastikan orientasi EXIF dari kamera smartphone dikoreksi menjadi tegak (upright)
    baik saat training, validasi lokal, maupun saat diuji oleh evaluate.py asisten.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            return ImageOps.exif_transpose(img)
        except Exception:
            return img


def _stabilize_linear_head(linear_layer: nn.Module, std: float = 0.01) -> None:
    """
    Menjaga skala logit awal tetap kecil (~0.0) agar CrossEntropyLoss pada Epoch 1
    tidak melonjak tinggi (mencegah representation shock pada backbone seperti EfficientNetV2-S).
    """
    if isinstance(linear_layer, nn.Linear):
        nn.init.normal_(linear_layer.weight, mean=0.0, std=std)
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)


def build_convnext_tiny(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    net = models.convnext_tiny(weights=weights)
    in_features = net.classifier[2].in_features
    net.classifier[2] = nn.Linear(in_features, num_classes)
    if pretrained:
        _stabilize_linear_head(net.classifier[2], std=0.01)
    return net


def build_deit3_small(
    model_name: str = MODEL_B_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
) -> nn.Module:
    net = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )
    if pretrained and hasattr(net, "head"):
        _stabilize_linear_head(net.head, std=0.01)
    return net


def build_swin_small(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> nn.Module:
    weights = models.Swin_S_Weights.DEFAULT if pretrained else None
    net = models.swin_s(weights=weights)
    in_features = net.head.in_features
    net.head = nn.Linear(in_features, num_classes)
    if pretrained:
        _stabilize_linear_head(net.head, std=0.01)
    return net


def build_efficientnet_v2_s(
    model_name: str = MODEL_D_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
) -> nn.Module:
    net = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )
    if pretrained and hasattr(net, "classifier"):
        _stabilize_linear_head(net.classifier, std=0.01)
    return net


class WasteHybridEnsembleWithTTA(nn.Module):
    """
    Quad-Paradigm Hybrid Ensemble (ConvNeXt-Tiny + DeiT3-Small + Swin-Small + EfficientNetV2-S)
    dalam 1 nn.Module.
    Mematuhi 100% kontrak evaluate.py asisten:
    - get_model() mengembalikan 1 objek nn.Module tanpa mengunduh bobot dari internet (pretrained=False)
    - Seluruh bobot keempat backbone + kalibrasi tersimpan di dalam 1 file submission/model.pth
    - forward(x) mengembalikan tensor [batch, 5]
    """

    def __init__(
        self,
        model_a_name: str = MODEL_A_NAME,
        model_b_name: str = MODEL_B_NAME,
        model_c_name: str = MODEL_C_NAME,
        model_d_name: str = MODEL_D_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = False,
    ):
        super().__init__()
        self.model_a = build_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
        self.model_b = build_deit3_small(
            model_name=model_b_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.model_c = build_swin_small(num_classes=num_classes, pretrained=pretrained)
        self.model_d = build_efficientnet_v2_s(
            model_name=model_d_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        # Bobot proporsi 4 model [w_a, w_b, w_c, w_d], temperature per model, bobot 6-view TTA, & logit bias
        self.register_buffer("ensemble_weights", torch.tensor([0.2500, 0.3000, 0.2500, 0.2000], dtype=torch.float32))
        self.register_buffer("model_temperatures", torch.tensor([1.0000, 1.0000, 1.0000, 1.0000], dtype=torch.float32))
        self.register_buffer("tta_view_weights", torch.tensor([0.2800, 0.2800, 0.1400, 0.1400, 0.0800, 0.0800], dtype=torch.float32))
        self.register_buffer("logit_adjustment", torch.zeros(num_classes, dtype=torch.float32))

    def _single_view_logits(self, x: torch.Tensor) -> torch.Tensor:
        w = self.ensemble_weights / (self.ensemble_weights.sum() + 1e-8)
        temps = torch.clamp(self.model_temperatures, min=0.25, max=4.0)
        blended = None

        for idx, net in enumerate((self.model_a, self.model_b, self.model_c, self.model_d)):
            if w[idx] > 1e-6:
                lp = F.log_softmax(net(x) / temps[idx], dim=-1)
                blended = w[idx] * lp if blended is None else blended + w[idx] * lp

        return blended + self.logit_adjustment.unsqueeze(0)

    @staticmethod
    def _center_rect_crop(x: torch.Tensor, ratio_h: float, ratio_w: float) -> torch.Tensor:
        _, _, h, w = x.shape
        ch = int(round(h * ratio_h))
        cw = int(round(w * ratio_w))
        top = (h - ch) // 2
        left = (w - cw) // 2
        cropped = x[:, :, top : top + ch, left : left + cw]
        return F.interpolate(cropped, size=(h, w), mode="bicubic", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_orig = self._single_view_logits(x)
        if not self.training:
            # 6-View Multi-Scale & Aspect-Rectified TTA:
            # 1. Original (224x224)
            # 2. Horizontal Flip
            # 3. Center Zoom 88% (fokus objek utama)
            # 4. Center Zoom 88% + Flip
            # 5. Aspect-Tall Rectification (96% H x 80% W -> mengoreksi botol/gelas plastik vertikal)
            # 6. Deep Texture Zoom 76% (fokus tekstur permukaan styrofoam vs gelas plastik vs saset)
            vw = self.tta_view_weights / (self.tta_view_weights.sum() + 1e-8)

            x_flip = torch.flip(x, dims=[-1])
            x_zoom88 = self._center_rect_crop(x, 0.88, 0.88)
            x_zoom88_flip = torch.flip(x_zoom88, dims=[-1])
            x_tall = self._center_rect_crop(x, 0.96, 0.80)
            x_zoom76 = self._center_rect_crop(x, 0.76, 0.76)

            out = vw[0] * logits_orig
            if vw[1] > 1e-6:
                out = out + vw[1] * self._single_view_logits(x_flip)
            if vw[2] > 1e-6:
                out = out + vw[2] * self._single_view_logits(x_zoom88)
            if vw[3] > 1e-6:
                out = out + vw[3] * self._single_view_logits(x_zoom88_flip)
            if vw[4] > 1e-6:
                out = out + vw[4] * self._single_view_logits(x_tall)
            if vw[5] > 1e-6:
                out = out + vw[5] * self._single_view_logits(x_zoom76)
            return out
        return logits_orig


def get_model() -> nn.Module:
    """
    Membangun arsitektur model saja (TANPA mengunduh bobot atau memuat model.pth).
    Output forward(x): Tensor logits berukuran [batch, 5].
    """
    return WasteHybridEnsembleWithTTA(
        model_a_name=MODEL_A_NAME,
        model_b_name=MODEL_B_NAME,
        model_c_name=MODEL_C_NAME,
        model_d_name=MODEL_D_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )


def get_transform() -> transforms.Compose:
    """
    Transformasi validasi dari PIL.Image (RGB) -> Tensor [3, 224, 224].
    Dilengkapi ExifTranspose() agar foto smartphone yang miring/terbalik otomatis diluruskan.
    """
    return transforms.Compose([
        ExifTranspose(),
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
