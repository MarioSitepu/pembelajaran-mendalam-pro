# Studio 2 — Klasifikasi Sampah · Kelompok <1>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

### 🔗 Tautan Model & Submission
- **Model Weights (`model.pth`)**: [https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.pth](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.pth)
- **Model Architecture & Preprocessing (`model.py`)**: [https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.py](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.py)

---

## Anggota Kelompok

| No  | Nama Lengkap            | NIM       | Kontribusi utama |
| --- | ----------------------- | --------- | ---------------- |
| 1   | George Haansraj         | 123140001 |                  |
| 2   | Mario Fransiskus Sitepu | 123140023 |                  |
| 3   |                         |           |                  |

## Ringkasan Pendekatan

Proyek ini menggunakan arsitektur **Quad-Paradigm Hybrid Super-Ensemble (@ 224x224)** yang mengombinasikan 4 model vision mutakhir (CNN modern dan Vision Transformer) ke dalam satu modul inferensi terpadu dengan integrasi **6-View Multi-Scale & Aspect-Rectified Test-Time Augmentation (TTA)** serta **ExifTranspose** otomatis untuk mengklasifikasikan gambar sampah ke dalam lima kelas:

1. `botol_plastik`
2. `gelas_plastik`
3. `kertas_kardus`
4. `saset_kemasan`
5. `styrofoam`

- **Arsitektur / backbone**:
  - **Backbone A (Modern 7x7 Depthwise CNN)**: `convnext_tiny` (`torchvision.models`, ~53.1 MB FP16)
  - **Backbone B (Global Self-Attention Vision Transformer)**: `deit3_small_patch16_224.fb_in22k_ft_in1k` (`timm`, ~41.4 MB FP16)
  - **Backbone C (Hierarchical Shifted-Window ViT)**: `swin_s` (`torchvision.models`, ~94.6 MB FP16)
  - **Backbone D (Fused-MBConv Fast CNN)**: `tf_efficientnetv2_s.in21k_ft_in1k` (`timm`, ~38.5 MB FP16)
- **Mekanisme Inferensi & TTA**:
  - **ExifTranspose**: Koreksi orientasi EXIF otomatis di dalam `get_transform()` agar foto smartphone yang rotasinya miring atau terbalik langsung diluruskan.
  - **Built-in 6-View Multi-Scale & Aspect-Rectified TTA**:
    1. *Original View* (224x224)
    2. *Horizontal Flip*
    3. *Center Zoom 88%* (fokus pada fitur inti objek sampah)
    4. *Center Zoom 88% + Horizontal Flip*
    5. *Aspect-Tall Rectification* (96% Height × 80% Width bicubic interpolation, mengoreksi distorsi rasio aspek botol dan gelas plastik vertikal)
    6. *Deep Texture Zoom 76%* (fokus menangkap diskriminasi tekstur permukaan styrofoam, saset, dan gelas plastik)
  - **Blended Calibration Buffers**: Menggunakan *registered buffers* untuk pembobotan ensemble terkalibrasi (`ensemble_weights`), *temperature scaling* per model (`model_temperatures`), bobot pandangan TTA (`tta_view_weights`), dan penyesuaian margin kelas (`logit_adjustment`).
  - **Kepatuhan Submission**: Seluruh bobot 4 backbone dikemas dalam satu berkas `submission/model.pth` terkompresi FP16 (~227.2 MB), sepenuhnya memenuhi batas ukuran (< 500 MB) dan kontrak fungsi `evaluate.py`.
- **Strategi pelatihan**: Transfer learning dengan inisialisasi bobot pretrained ImageNet, stabilisasi classifier head (`_stabilize_linear_head` dengan normal std=0.01) guna mencegah representational shock, regularisasi drop rate & drop path rate, serta kalibrasi temperatur logit.
- **Data tambahan**: Tidak ada (hanya menggunakan dataset resmi yang disediakan untuk kompetisi).
- **Ukuran input & preprocessing**: 224x224 (ExifTranspose + Bicubic Resize 224x224 antialias + ToTensor + Normalisasi ImageNet: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`).

## Pretrained Model dan Sumber Data

- **Arsitektur pretrained:**
  - `convnext_tiny` (Pretrained ImageNet-1K, Torchvision)
  - `deit3_small_patch16_224.fb_in22k_ft_in1k` (Pretrained ImageNet-22K fine-tuned 1K, timm)
  - `swin_s` (Pretrained ImageNet-1K, Torchvision)
  - `tf_efficientnetv2_s.in21k_ft_in1k` (Pretrained ImageNet-21K fine-tuned 1K, timm)
- **Dataset:** Dataset resmi kompetisi Studio 2.
- **Data tambahan:** Tidak ada.

## Hasil

| Putaran            | Commit    | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
| ------------------ | --------- | ------------------------------- | ---------------------------- |
| Preview 1 (14 Okt) | `50e8a37` | 97,83%                          | 95,25%                       |
| Preview 2 (21 Okt) | `055ac11` | 98,90%                          |                              |
| Final (25 Okt)     |           |                                 |                              |

### Hasil Validasi Lokal

#### Preview 2 (21 Okt — Quad-Paradigm Super-Ensemble)

| Metrik | Hasil |
|---|---:|
| Best Validation Macro-F1 | 98,90% (0.9890) |
| Validation Accuracy | 98,91% (545/551 gambar benar) |
| Jumlah gambar validasi | 551 |

*Catatan: Skor baseline kompetisi ResNet-18 = 0.9200 (92,00%).*

#### Preview 1 (14 Okt — Baseline ConvNeXt-Tiny)

| Metrik | Hasil |
|---|---:|
| Best Validation Macro-F1 | 97,83% (Asistensi: 95,25%) |
| Validation Accuracy | 97,82% (Asistensi: 95,28%) |
| Jumlah gambar validasi | 551 |

*Catatan: hasil di atas merupakan hasil validasi lokal, bukan hasil hidden validation set kompetisi.*

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- **Eksperimen Backbone Tunggal (Preview 1)**: ConvNeXt-Tiny menunjukkan konvergensi awal yang kuat dengan Macro-F1 validasi lokal 97,83% (akurasi 97,82%) serta skor resmi asisten Macro-F1 95,25% (akurasi 95,28%).
- **Transisi ke Quad-Paradigm Hybrid Ensemble (Preview 2)**: Menggabungkan keunggulan representasi lokal dari CNN (`convnext_tiny` & `tf_efficientnetv2_s`) dengan representasi global transformer (`deit3_small` & `swin_s`) melonjakkan Macro-F1 validasi lokal ke 98,90% dan akurasi 98,91% (545/551 benar).
- **Penanganan Orientasi Kamera (ExifTranspose)**: Integrasi transformasi EXIF otomatis meniadakan penurunan akurasi pada gambar smartphone yang tersimpan dalam rotasi landscape/portrait metadata.
- **Aspect-Rectified Test-Time Augmentation (TTA)**: Penggunaan multi-view TTA (terutama aspect-tall crop dan texture-zoom) meningkatkan ketahanan prediksi terhadap variasi framing dan skala objek di lapangan.
- **Optimasi Bobot FP16**: Seluruh bobot keempat model berhasil dikompilasi ke dalam format FP16 sebesar ~227.2 MB, jauh di bawah batas maksimum 500 MB.

## Cara Reproduksi Evaluasi

Pastikan Python dan dependensi yang diperlukan (`torch`, `torchvision`, `timm`, `Pillow`) telah terpasang, serta data validasi tersedia dalam struktur folder sesuai nama kelas yang ditetapkan.

```bash
python evaluate.py --submission ./submission --data ./my_val
```

Ganti `./my_val` dengan direktori validation set lokal.

## Struktur Repositori

```
submission/
  model.py          # mendefinisikan get_model() dan get_transform() sesuai kontrak evaluator
  model.pth         # bobot model hasil fine-tuning, disimpan menggunakan Git LFS
  requirements.txt  # dependensi tambahan (timm)
evaluate.py         # salinan skrip evaluasi resmi
```

Catatan berkas dan tautan:
- [`submission/model.py`](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.py): Mendefinisikan arsitektur `WasteHybridEnsembleWithTTA` melalui fungsi `get_model()` dan preprocessing `get_transform()` sesuai kontrak resmi evaluator.
- [`submission/model.pth`](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.pth): State dictionary lengkap berukuran ~227.2 MB (FP16) berisi seluruh bobot 4 model backbone serta calibration buffer, dilacak menggunakan Git LFS.
- `submission/requirements.txt`: Mendaftarkan dependensi pihak ketiga (`timm`) agar otomatis dipasang di environment evaluasi asisten.

## Teknologi

- Python
- PyTorch
- Torchvision
- timm (PyTorch Image Models)
- Pillow
- Git LFS

