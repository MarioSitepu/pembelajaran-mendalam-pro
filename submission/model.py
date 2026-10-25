"""
submission/model.py
Kontrak Resmi Studio Proyek 2 (IF25-40401 Deep Learning ITERA)
Arsitektur V6.2: Hybrid 3-Member Full ImageNet-22K Super-Ensemble + Context-Preserving 4-View TTA
- Member A (Modern 7x7 Depthwise CNN) : timm convnext_tiny.fb_in22k_ft_in1k (~53.1 MB FP16)
- Member B (ConvNeXt-Tiny LLRD 0.85)  : timm convnext_tiny.fb_in22k_ft_in1k (DropPath 0.10) (~53.1 MB FP16)
- Member C (Swin-Small LLRD 0.90)     : timm swin_small_patch4_window7_224.ms_in22k_ft_in1k (~94.6 MB FP16)
- Geometri Preprocessing Konteks Penuh: ExifTranspose() -> Resize(256) -> CenterCrop(256)
- Context-Preserving 4-View Softmax TTA:
  * View 0 & 1: Exact Slice 224x224 Tengah (x[:, :, 16:240, 16:240]) + Flip (Ketajaman Tekstur 100%, Nol Blur)
  * View 2 & 3: Antialiased Full-Frame Downscale 256->224 + Flip (Konteks Tepi Wadah Styrofoam/Saset 100% Utuh)
- Total ukuran submission/model.pth (FP16) ~199.5 MB (jauh di bawah batas kuota 500 MB evaluate.py)
"""

from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

MODEL_A_NAME = "convnext_tiny.fb_in22k_ft_in1k"
MODEL_B_NAME = "convnext_tiny.fb_in22k_ft_in1k"
MODEL_C_NAME = "swin_small_patch4_window7_224.ms_in22k_ft_in1k"
NUM_CLASSES = 5
IMG_SIZE = 224
EVAL_CROP_SIZE = 256

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


def build_convnext_tiny(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    return timm.create_model(
        MODEL_A_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


def build_swin_small(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    drop_path_rate: float = 0.15,
) -> nn.Module:
    return timm.create_model(
        MODEL_C_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


class WasteHybridEnsembleWithTTA(nn.Module):
    """
    Hybrid 3-Member Full ImageNet-22K Ensemble (2x ConvNeXt-Tiny + 1x Swin-Small) dalam 1 nn.Module.
    Mematuhi 100% kontrak evaluate.py asisten:
    - get_model() mengembalikan 1 objek nn.Module tanpa mengunduh bobot dari internet (pretrained=False)
    - Seluruh bobot ketiga backbone + kalibrasi tersimpan di dalam 1 file submission/model.pth
    - forward(x) mengembalikan tensor [batch, 5]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = False,
    ):
        super().__init__()
        self.model_a = build_convnext_tiny(num_classes=num_classes, pretrained=pretrained, drop_path_rate=0.0)
        self.model_b = build_convnext_tiny(num_classes=num_classes, pretrained=pretrained, drop_path_rate=0.10)
        self.model_c = build_swin_small(num_classes=num_classes, pretrained=pretrained, drop_path_rate=0.15)

        # Buffer proporsi 3 model [w_a, w_b, w_c], suhu kalibrasi NLL, bobot 4-view TTA & bias
        self.register_buffer("ensemble_weights", torch.tensor([0.3, 0.46, 0.24], dtype=torch.float32))
        self.register_buffer("model_temperatures", torch.tensor([0.75, 0.75, 0.75], dtype=torch.float32))
        self.register_buffer("tta_view_weights", torch.tensor([0.35, 0.35, 0.15, 0.15], dtype=torch.float32))
        self.register_buffer("logit_adjustment", torch.zeros(num_classes, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flexible Context-Preserving Softmax Inference (Supports No-TTA, 2-View, & 4-View):
        - Jika input berukuran 256x256 (dari get_transform()):
          * View 0: Slice eksak 224x224 di tengah (16:240, 16:240) -> 100% identik dengan CenterCrop 224 murni
          * View 1: Flip horizontal dari slice tengah (aktif HANYA jika vw[1] > 0)
          * View 2: Antialiased downscale 256->224 (aktif HANYA jika vw[2] > 0)
          * View 3: Flip horizontal dari downscale 256->224 (aktif HANYA jika vw[3] > 0)
        - Mendukung Dual-Ensemble (jika w_C == 0, Model C Swin-Small otomatis di-skip)
        """
        w = self.ensemble_weights / (self.ensemble_weights.sum() + 1e-8)
        vw = self.tta_view_weights / (self.tta_view_weights.sum() + 1e-8)
        t = self.model_temperatures

        h, wid = x.shape[-2], x.shape[-1]
        use_crop = bool(vw[0] > 1e-5 or vw[1] > 1e-5)
        use_flip_crop = bool(vw[1] > 1e-5)
        use_full = bool(vw[2] > 1e-5 or vw[3] > 1e-5)
        use_flip_full = bool(vw[3] > 1e-5)

        x_c, x_cf, x_f, x_ff = None, None, None, None

        if h == EVAL_CROP_SIZE and wid == EVAL_CROP_SIZE:
            margin = (EVAL_CROP_SIZE - IMG_SIZE) // 2  # 16 px
            if use_crop:
                x_c = x[:, :, margin : margin + IMG_SIZE, margin : margin + IMG_SIZE]
                if use_flip_crop:
                    x_cf = torch.flip(x_c, dims=[-1])
            if use_full:
                x_f = F.interpolate(
                    x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False, antialias=True
                )
                if use_flip_full:
                    x_ff = torch.flip(x_f, dims=[-1])
        else:
            x_c = x if (h == IMG_SIZE and wid == IMG_SIZE) else F.interpolate(
                x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False, antialias=True
            )
            if use_flip_crop:
                x_cf = torch.flip(x_c, dims=[-1])
            use_full = False

        def _member_probs(net: nn.Module, temp: torch.Tensor) -> torch.Tensor:
            p = torch.zeros(x.shape[0], NUM_CLASSES, device=x.device, dtype=torch.float32)
            if x_c is not None and vw[0] > 1e-5:
                p = p + vw[0] * (net(x_c) / temp).softmax(dim=-1)
            if x_cf is not None and vw[1] > 1e-5:
                p = p + vw[1] * (net(x_cf) / temp).softmax(dim=-1)
            if use_full:
                if x_f is not None and vw[2] > 1e-5:
                    p = p + vw[2] * (net(x_f) / temp).softmax(dim=-1)
                if x_ff is not None and vw[3] > 1e-5:
                    p = p + vw[3] * (net(x_ff) / temp).softmax(dim=-1)
            return p

        probs = torch.zeros(x.shape[0], NUM_CLASSES, device=x.device, dtype=torch.float32)
        if w[0] > 1e-5:
            probs = probs + w[0] * _member_probs(self.model_a, t[0])
        if w[1] > 1e-5:
            probs = probs + w[1] * _member_probs(self.model_b, t[1])
        if w[2] > 1e-5:
            probs = probs + w[2] * _member_probs(self.model_c, t[2])
        return probs.clamp_min(1e-12).log()


WasteHybridEnsembleV6 = WasteHybridEnsembleWithTTA


def get_model() -> nn.Module:
    """
    Membangun arsitektur model saja (TANPA mengunduh bobot atau memuat model.pth).
    """
    return WasteHybridEnsembleWithTTA(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )


def get_transform() -> transforms.Compose:
    """
    Transformasi validasi / inferensi dari PIL.Image (RGB) -> Tensor [3, 256, 256].
    Geometri Preprocessing: ExifTranspose() -> Resize(256) -> CenterCrop(256).
    Di dalam forward(), slice tengah [16:240, 16:240] menghasilkan CenterCrop(224) murni
    (nol blur), sementara seluruh kanvas 256x256 memberikan konteks tepi objek 100% utuh.
    """
    return transforms.Compose([
        ExifTranspose(),
        transforms.Resize(EVAL_CROP_SIZE),
        transforms.CenterCrop(EVAL_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
