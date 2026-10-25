# Studio 2 — Klasifikasi Sampah · Kelompok Crookie

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

### 🔗 Tautan Model & Submission
- **Model Weights (`model.pth`)**: [https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.pth](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.pth)
- **Model Architecture & Preprocessing (`model.py`)**: [https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.py](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.py)

---

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | George Haansraj | 123140001 | Riset Arsitektur Vision Transformer (Swin-Small), LLRD 0.90, Greedy Model Soup & Analisis Generalisasi |
| 2  | Mario Fransiskus Sitepu | 123140023 | Rekayasa Pipeline Data, dHash 256-bit + Union-Find, Context-Preserving 4-View TTA & Evaluasi Resmi |

---

## Ringkasan Pendekatan

- **Arsitektur / backbone**: **Hybrid 3-Member Full ImageNet-22K Super-Ensemble** dalam satu modul inferensi terpadu:
  1. *Anggota 1*: `convnext_tiny.fb_in22k_ft_in1k` (Pretrained Meta ImageNet-22K $\rightarrow$ 1K, Modern 7×7 Depthwise Separable CNN, ~53.1 MB FP16).
  2. *Anggota 2*: `convnext_tiny.fb_in22k_ft_in1k` (Stochastic Depth `drop_path_rate = 0.10`, Layer-wise Learning Rate Decay `0.85`, ~53.1 MB FP16).
  3. *Anggota 3*: `swin_small_patch4_window7_224.ms_in22k_ft_in1k` (Hierarchical Shifted-Window Vision Transformer, `drop_path_rate = 0.15`, LLRD `0.90`, ~94.6 MB FP16).
  - Seluruh bobot dikemas dalam satu checkpoint FP16 `submission/model.pth` (**199.46 MB**, batas kuota `< 500 MB`).
  - Fusi probabilitas: Rata-rata aritmatika Softmax Mean terkalibrasi $\sum w_m \text{softmax}(z_m / T_m)$ dengan bobot simplex $[0.30, 0.46, 0.24]$ dan suhu NLL stabil $T = [0.75, 0.75, 0.75]$ guna mengeliminasi penalti efek veto log-softmax.
- **Strategi pelatihan**:
  - *Head Warmup*: 2 epoch membekukan backbone (`.eval()`), hanya melatih classifier head (`lr = 1e-3`) agar inisialisasi awal head tidak merusak representasi pretrained.
  - *Full Fine-Tuning*: 20 epoch dengan optimizer AdamW (`weight_decay = 0.05` khusus tensor matriks 2D), Cosine Annealing scheduler, dan Layer-wise Learning Rate Decay (LLRD) yang mengecilkan LR secara bertingkat ke arah stem.
  - *Regularisasi Anti-Oklusi*: Penerapan batch `CutMix (alpha=1.0)` & `Mixup (alpha=0.2)` pada 50% batch, `TrivialAugmentWide()`, `RandomErasing (p=0.25)`, dan `label_smoothing = 0.10`.
  - *Greedy Model Soup*: Merata-ratakan bobot checkpoint berkinerja tinggi (*flat minima*) dalam lembah konvergensi yang sama untuk memperkuat generalisasi data uji baru tanpa *blind refitting*.
- **Data tambahan**: Tidak ada (hanya menggunakan dataset resmi yang disediakan untuk kompetisi, dipartisi dengan 256-bit dHash + Union-Find bebas leakage 2.343 Train : 410 Val).
- **Ukuran input & preprocessing**:
  - *Training*: `ExifTranspose` $\rightarrow$ `RandomResizedCrop(224, scale=(0.5, 1.0))` $\rightarrow$ `RandomHorizontalFlip` $\rightarrow$ `ColorJitter` $\rightarrow$ `TrivialAugmentWide` $\rightarrow$ `ToTensor` $\rightarrow$ `Normalize(ImageNet)` $\rightarrow$ `RandomErasing(p=0.25)`.
  - *Evaluasi / Validasi (`get_transform`)*: `ExifTranspose()` $\rightarrow$ `Resize(256, interpolation=BICUBIC, antialias=True)` $\rightarrow$ `CenterCrop(256)` $\rightarrow$ `ToTensor()` $\rightarrow$ `Normalize(ImageNet)`.
  - *Inferensi Context-Preserving 4-View TTA*: Mengombinasikan potongan eksak tengah 224×224 (ketajaman tekstur 100%, nol blur interpolasi) dengan antialiased downscale 256 $\rightarrow$ 224 (konteks batas wadah sampah 100% utuh), serta mendukung input langsung 224×224 tanpa kegagalan eksekusi.

---

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `50e8a37` | 0.9783 (97,83%) | 0.9525 (95,25%) |
| Preview 2 (21 Okt) | `055ac11` | 0.9890 (98,90%) | 0.9533 (95,33%) |
| Final (25 Okt)     | `b0d35e9` | **0.9832 (98,32%)** | *(Menunggu Pengumuman)* |

### Rincian Evaluasi Final (25 Okt) pada Split Bebas Leakage (410 Citra)
- **Akurasi Keseluruhan**: **`0.9829` (`98.29%`)** (403 benar dari 410 citra, hanya 7 salah).
- **Waktu Inferensi Evaluasi**: **`11.63` detik** | **Ukuran Checkpoint**: **`199.46` MB** (FP16).
- **Performa per Kategori Sampah**:
  - `botol_plastik`: Precision `1.0000` · Recall `1.0000` · **F1: `1.0000` (Sempurna: 78/78, 0 FP)**
  - `gelas_plastik`: Precision `0.9512` · Recall `1.0000` · **F1: `0.9750` (78/78 recall)**
  - `kertas_kardus`: Precision `0.9643` · Recall `1.0000` · **F1: `0.9818` (54/54 recall)**
  - `saset_kemasan`: Precision `0.9897` · Recall `0.9600` · **F1: `0.9746` (96/100 benar)**
  - `styrofoam`: Precision `1.0000` · Recall `0.9700` · **F1: `0.9848` (97/100 benar)**

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- **Preview 1 (14 Okt — Single Backbone Baseline)**: Eksperimen model tunggal ConvNeXt-Tiny ImageNet-1K. Meraih Macro-F1 validasi lokal 97,83% dan skor asisten 95,25%. Berhasil membentuk baseline cepat, namun masih rentan terhadap variasi rotasi dan oklusi kemasan.
- **Preview 2 (21 Okt — Quad-Ensemble & Identifikasi Gap Evaluasi)**: Menggabungkan 4 model (ConvNeXt, DeiT3, Swin-S, EfficientNetV2). Macro-F1 lokal menyentuh 98,90%, tetapi evaluasi asisten turun ke 95,33%. Audit forensik mengungkap 3 masalah utama: (1) kebocoran near-duplicate akibat partisi SHA-256 sederhana, (2) distorsi aspek rasio dan blur akibat in-tensor crop upsampling TTA (76% & 88%), dan (3) penalti efek veto log-softmax serta kapasitas pretraining ImageNet-1K yang terbatas.
- **Final (25 Okt — Super-Ensemble V6.3 Bebas Leakage & Context TTA)**:
  - *Yang berhasil*:
    1. **Partisi 256-bit dHash + Union-Find** (jarak Hamming $\le 30$) melenyapkan kebocoran near-duplicate, menghasilkan validasi murni 85:15 yang jujur dan bebas inflasi skor.
    2. **Fusi Heterogen 3-Member Full ImageNet-22K** (2x ConvNeXt-Tiny + 1x Swin-Small) dengan Greedy Model Soup menyatukan representasi lokal CNN dan relasional global Vision Transformer ke permukaan loss datar (*flat minima*).
    3. **Context-Preserving 4-View TTA** (slice 224 nol blur + downscale kanvas 256) mengeliminasi blur tekstur mikro sekaligus menjaga kontur tepi wadah, membawa F1 `botol_plastik` ke 1.0000 sempurna dan `kertas_kardus` ke 0.9818.
    4. **Softmax Mean & Bounded Temperature Scaling ($T=0.75$)** membuang efek veto log-softmax dan menghentikan overconfidence false positive.
  - *Yang gagal*: In-tensor upsampling crop TTA terdahulu (mengaburkan tekstur mikro pori-pori styrofoam), blind full-data refit di akhir (meningkatkan risiko overfit data lokal), serta suhu overconfident $T < 0.70$ (memicu salah tebak dominan).

---

## Cara Reproduksi

```bash
# 1. Pasang dependensi
pip install torch torchvision timm Pillow

# 2. Perintah melatih model end-to-end (Data Split, Model Soup, TTA, Kalibrasi)
python notebooks/cnn-sampah.py

# 3. Pengujian resmi kontrak evaluate.py
python evaluate.py --submission ./submission --data ./my_val
```

---

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() sesuai kontrak evaluator
  model.pth         # state_dict Super-Ensemble terbaik (199.46 MB FP16)
  requirements.txt  # paket tambahan evaluasi (timm)
evaluate.py         # salinan skrip evaluasi resmi dosen/asisten
notebooks/
  cnn-sampah.py     # kode pipeline pelatihan lengkap end-to-end
  common.py         # utilitas dHash, Union-Find, augmentasi, dan LLRD
diagnostics/        # grafik konvergensi, confusion matrix, dan log validasi
README.md           # dokumentasi resmi proyek
```
