# Studio 2 — Klasifikasi Sampah · Kelompok Crookie

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

### 🔗 Tautan Model & Submission
- **Model Weights (`model.pth`)**: [https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.pth](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.pth)
- **Model Architecture & Preprocessing (`model.py`)**: [https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.py](https://github.com/MarioSitepu/pembelajaran-mendalam-pro/blob/main/submission/model.py)

---

## Anggota Kelompok

| No  | Nama Lengkap            | NIM       | Kontribusi Utama |
| --- | ----------------------- | --------- | ---------------- |
| 1   | George Haansraj         | 123140001 | Riset Arsitektur Vision Transformer (Swin-Small), LLRD 0.90, Greedy Model Soup & Analisis Generalisasi |
| 2   | Mario Fransiskus Sitepu | 123140023 | Rekayasa Pipeline Data, dHash 256-bit + Union-Find, Context-Preserving 4-View TTA & Evaluasi Resmi |

---

## 1. Ringkasan Eksekutif & Hasil Evaluasi Resmi

Kelompok Crookie mengembangkan sistem klasifikasi sampah cerdas berbasis **Hybrid 3-Member Full ImageNet-22K Super-Ensemble** yang mengintegrasikan keunggulan representasi lokal Modern Depthwise CNN (*ConvNeXt-Tiny*) dan representasi relasional global Hierarchical Vision Transformer (*Swin-Small*). Pipeline ini dilengkapi optimasi generalisasi **Greedy Model Soup** serta mekanisme inferensi **Context-Preserving 4-View TTA**.

Berdasarkan pengujian resmi menggunakan skrip kontrak asisten dosen `evaluate.py` pada split validasi lokal murni bebas kebocoran (*zero data leakage*, 410 citra), model mencapai performa tingkat tertinggi:

| Metrik Evaluasi Resmi (`evaluate.py`) | Hasil Capaian Kelompok Crookie | Keterangan & Standar Penilaian |
| :--- | :---: | :--- |
| **Macro-F1 Score** | **`0.9832` (`98.32%`)** | Metrik utama penilaian kompetisi |
| **Akurasi Keseluruhan** | **`0.9829` (`98.29%`)** | **403 dari 410** citra validasi berhasil diklasifikasikan dengan benar |
| **Total Kesalahan Prediksi** | **Hanya 7 citra** | Terisolasi pada kasus batas kemasan komposit |
| **Ukuran File `submission/model.pth`** | **`199.46` MB (FP16)** | Jauh di bawah batas kuota penilaian (`500 MB`) |
| **Waktu Inferensi Evaluasi** | **`11.63` detik** | Sangat cepat untuk 410 citra (batas `< 900` detik) |
| **Kepatuhan Kontrak Kode** | **100% Patuh & Lolos** | `get_model()` dan `get_transform()` kompatibel penuh |

### Performa Individual Anggota Ensemble pada Validasi Lokal (`my_val` = 410 Citra)

| Anggota Model | Arsitektur Backbone | Strategi Optimasi & Generalisasi | Val Acc | Val Macro-F1 |
| :--- | :--- | :--- | :---: | :---: |
| **Anggota 1** | `convnext_tiny.fb_in22k_ft_in1k` | Standard Fine-Tuning + Greedy Top-4 Model Soup | `0.9805` | **`0.9804`** |
| **Anggota 2** | `convnext_tiny.fb_in22k_ft_in1k` | LLRD 0.85 + DropPath 0.10 + Greedy Top-3 Model Soup | `0.9756` | **`0.9735`** |
| **Anggota 3** | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | LLRD 0.90 + DropPath 0.15 + Greedy Top-2 Model Soup | `0.9707` | **`0.9716`** |
| **Super-Ensemble** | **Heterogeneous 3-Member Super-Ensemble** | **Simplex Calibrated ($[0.30, 0.46, 0.24]$) + 4-View TTA** | **`0.9829`** | **`0.9832`** |

---

## 2. Prinsip Rekayasa Metodologi Kelompok Crookie

Untuk memastikan model memiliki daya generalisasi yang kokoh pada *hidden test set* evaluasi dosen tanpa risiko *overfitting*, Kelompok Crookie menerapkan lima pilar rekayasa metodologi:

### A. Partisi Dataset Bebas Kebocoran Mutlak (*Zero Data Leakage Group Split*)
Dataset sampah dunia nyata yang dikumpulkan dari katalog atau web mengandung banyak foto beruntun (*burst shots*), variasi sudut pengambilan, dan kompresi ulang dari objek yang sama (*near-duplicate*).
* Algoritma partisi standar yang hanya mengandalkan *hash* byte (seperti SHA-256) memecah gambar kembar ke dalam data latih dan validasi, menyebabkan inflasi skor semu.
* **Solusi Crookie**: Menerapkan **256-bit Perceptual Difference Hash (`dHash`)** dengan jarak Hamming $\le 30$ bit yang digabungkan dengan **Union-Find Graph Connected Components**. Seluruh kelompok citra yang serupa dikunci ke dalam satu partisi yang sama sebelum dilakukan pembagian berstrata 85% Train (2.343 citra) : 15% Val (410 citra). Hasilnya, data validasi 100% independen dan mencerminkan distribusi data uji yang sesungguhnya.

### B. Pemanfaatan Representasi Skala Besar (Full `ImageNet-22K`)
* Seluruh *backbone* diinisialisasi dari bobot pra-latih **ImageNet-22K (14,2 juta citra, 21.841 kelas)** sebelum diadaptasikan ke ImageNet-1K (`timm`).
* Ruang fitur berdimensi luas ini memberikan pemahaman semantik yang mendalam terhadap berbagai variasi tekstur bahan plastik, logo merek kemasan saset, serat kertas, serta kondisi pencahayaan ekstrem.

### C. Regularisasi Berlapis Anti-Oklusi & Anti-Menghafal
Mencegah jaringan menghafal pola latar belakang foto atau teks toko melalui 4 instrumen regularisasi simultan:
1. **Batch `CutMix (alpha=1.0)` & `Mixup (alpha=0.2)` dengan probabilitas `p=0.5`**: Melatih model mengenali material sampah yang saling bertumpuk atau terpotong sebagian.
2. **`TrivialAugmentWide()`**: Memberikan eksplorasi variasi fotometrik dan geometrik yang luas tanpa distorsi label.
3. **`RandomErasing (p=0.25)`**: Menghapus blok piksel secara acak agar model tidak bergantung pada watermark atau logo tertentu.
4. **`Label Smoothing (0.10)` & Weight Decay Terpisah**: Mengurangi overconfidence probabilitas dan menjaga stabilitas parameter bobot.

### D. Geometri Konteks Penuh (*Context-Preserving 4-View TTA*)
Metode TTA konvensional yang memotong bagian tengah lalu melakukan pembesaran ulang (*in-tensor upsampling*) terbukti mengaburkan detail pori-pori styrofoam dan membuang informasi batas wadah.
* **Solusi Crookie**: Preprocessing memelihara kanvas `256x256` (`Resize(256) -> CenterCrop(256)`). Pada tahap inferensi `forward(x)`:
  - *View 0 & 1*: Potongan eksak tengah `224x224` (`x[:, :, 16:240, 16:240]`) + Horizontal Flip $\rightarrow$ **Ketajaman tekstur 100%, nol blur interpolasi**.
  - *View 2 & 3*: *Antialiased downscale* `256 -> 224` + Horizontal Flip $\rightarrow$ **Konteks batas wadah styrofoam/kemasan 100% utuh**.
  - *Universal Fallback*: Jika dievaluasi oleh skrip eksternal yang memberikan tensor `224x224`, jaringan secara otomatis beradaptasi tanpa kegagalan eksekusi.

### E. Greedy Model Soup & Kalibrasi Probabilitas Sehat
* **Greedy Model Soup**: Alih-alih hanya mengandalkan checkpoint tunggal terakhir yang rentan terjebak pada *sharp minima*, Crookie menerapkan rata-rata bobot (*weight averaging*) terhadap checkpoint-checkpoint terbaik di lembah konvergensi yang sama. Secara teoritis, ini menghasilkan permukaan loss datar (*flat minima*) yang terbukti paling tangguh terhadap pergeseran distribusi data baru.
* **Softmax Mean & Bounded Temperature**: Prediksi ensemble digabungkan melalui rata-rata aritmatika probabilitas terkalibrasi $\sum w_m \text{softmax}(z_m / T_m)$ dengan pencarian suhu sehat $T \in [0.75, 1.30]$. Hal ini sepenuhnya mengeliminasi efek veto log-softmax maupun overconfidence false positive.

---

## 3. Spesifikasi Teknis Pipeline Super-Ensemble

### A. Konfigurasi 3 Anggota Ensemble
1. **Anggota 1 (`convnext_tiny.fb_in22k_ft_in1k`)**:
   - Arsitektur: 7x7 Depthwise Separable Convolutional Network
   - Pelatihan: 2 epoch Frozen Head Warmup (`lr = 1e-3`) + 20 epoch Fine-Tuning (`backbone_lr = 1e-4`, `head_lr = 1e-3`)
   - Bobot Akhir: Greedy Top-4 Model Soup (Macro-F1: `0.9804`)
2. **Anggota 2 (`convnext_tiny.fb_in22k_ft_in1k`)**:
   - Regularisasi: Stochastic Depth (`drop_path_rate = 0.10`)
   - Optimasi: Layer-wise Learning Rate Decay (`layer_decay = 0.85`, 12 tahapan blok)
   - Bobot Akhir: Greedy Top-3 Model Soup (Macro-F1: `0.9735`)
3. **Anggota 3 (`swin_small_patch4_window7_224.ms_in22k_ft_in1k`)**:
   - Arsitektur: Hierarchical Vision Transformer with Shifted Windows (`drop_path_rate = 0.15`)
   - Optimasi: Layer-wise Learning Rate Decay (`layer_decay = 0.90`, 24 blok), `base_lr = 1.2e-4`
   - Bobot Akhir: Greedy Top-2 Model Soup (Macro-F1: `0.9716`)

### B. Parameter Kalibrasi Final pada Checkpoint `submission/model.pth`
- **Bobot Proporsi Ensemble ($w_A, w_B, w_C$)**: `[0.30, 0.46, 0.24]`
- **Suhu Skalasi Logit ($T_A, T_B, T_C$)**: `[0.75, 0.75, 0.75]`
- **Bobot 4-View TTA ($v_{\text{crop}}, v_{\text{flip\_crop}}, v_{\text{full}}, v_{\text{flip\_full}}$)**: `[0.35, 0.35, 0.15, 0.15]`
- **Format Penyimpanan**: Float16 State Dict terkompresi (`199.46 MB`).

### C. Multi-Platform Auto-Resolver & Google Drive Fallback
Skrip terintegrasi secara cerdas mengenali lingkungan eksekusi:
- Kaggle Input (`/kaggle/input/...`)
- Google Colab (`/content/...` dan auto-mount Google Drive)
- Direktori lokal (`./train`)
- **Unduh Otomatis**: Jika dataset lokal tidak tersedia, skrip secara otomatis mengunduh dan mengekstrak dataset resmi dari Google Drive Folder ID: `1Koypcv5vu1LD27wb5WOQKkGM7ZSD9VF8`.

---

## 4. Laporan Rinci Evaluasi Resmi (`evaluate.py`)

Pengujian menyeluruh menggunakan modul resmi `evaluate.py` pada 410 citra validasi lokal menghasilkan metrik presisi tinggi di setiap kelas:

```text
========================================================================================
                            UJI RESMI SUBMISSION EVALUATE.PY                            
========================================================================================
Total Sampel      : 410 gambar | Waktu Evaluasi: 11.63 detik
Akurasi Rata-rata : 0.9829 (98.29%)
Macro-F1 Score    : 0.9832 (98.32%)
----------------------------------------------------------------------------------------
Kelas                  Prec   Recall       F1     Spec   Support
----------------------------------------------------------------------------------------
botol_plastik        1.0000   1.0000   1.0000   1.0000        78
gelas_plastik        0.9512   1.0000   0.9750   0.9880        78
kertas_kardus        0.9643   1.0000   0.9818   0.9944        54
saset_kemasan        0.9897   0.9600   0.9746   0.9968       100
styrofoam            1.0000   0.9700   0.9848   1.0000       100
----------------------------------------------------------------------------------------
Confusion Matrix (Baris: Aktual, Kolom: Prediksi):
           botol_  gelas_  kertas  saset_  styrof
botol_pla      78       0       0       0       0
gelas_pla       0      78       0       0       0
kertas_ka       0       0      54       0       0
saset_kem       0       2       2      96       0
styrofoam       0       2       0       1      97
========================================================================================
```

### Analisis Kinerja per Kategori Sampah
1. **`botol_plastik` (F1-Score: 1.0000 / Sempurna)**:
   Seluruh 78 sampel berhasil dikenali dengan sempurna tanpa ada kesalahan deteksi maupun *false positive* dari kelas lain.
2. **`gelas_plastik` (Recall: 100%, F1-Score: 0.9750)**:
   Semua 78 gelas plastik teridentifikasi lengkap. Terdapat 4 *false positive* dari objek hibrida (2 saset transparan dan 2 cup/cangkir styrofoam).
3. **`kertas_kardus` (Recall: 100%, F1-Score: 0.9818)**:
   Seluruh 54 sampel kardus dan kertas terdeteksi tanpa luput (*zero false negative*).
4. **`saset_kemasan` (F1-Score: 0.9746)**:
   Berhasil mengidentifikasi 96 dari 100 sampel kemasan biskuit, kopi, dan deterjen berlogo.
5. **`styrofoam` (Precision: 100%, F1-Score: 0.9848)**:
   Mencapai presisi sempurna 1.0000 di mana tidak ada benda sampah lain yang salah dikira sebagai styrofoam. 3 kesalahan terjadi pada kemasan cangkir mie instan/kopi styrofoam yang memiliki bentuk menyerupai gelas.

---

## 5. Panduan Reproduksi & Evaluasi Cepat

### Persiapan Dependensi Lingkungan
```bash
pip install torch torchvision timm Pillow
```

### Menjalankan Pipeline Pelatihan End-to-End
Skrip dapat dijalankan langsung di lingkungan lokal, Kaggle, maupun Google Colab:
```bash
python notebooks/cnn-sampah.py
```
*Pipeline akan memvalidasi split data, melatih/memuat model soup, mengoptimasi bobot kalibrasi, serta memperbarui file `submission/model.py` dan `submission/model.pth` secara otomatis.*

### Menjalankan Pengujian Resmi Standar Asisten Dosen
Untuk memverifikasi kepatuhan kontrak dan skor evaluasi:
```bash
python evaluate.py --submission ./submission --data ./my_val
```

*Skrip resmi akan memuat fungsi `get_model()` dan `get_transform()` dari `submission/model.py`, membaca tensor bobot FP16 dari `submission/model.pth`, dan menyajikan laporan akurasi serta Macro-F1 resmi.*

---

## 6. Struktur Repositori

```text
├── train/                               # Dataset mentah resmi (5 kelas sampah)
├── my_train/                            # Split latih bebas kebocoran (2.343 citra, 85.1%)
├── my_val/                              # Split validasi resmi bebas kebocoran (410 citra, 14.9%)
├── submission/
│   ├── model.py                         # Kontrak evaluasi resmi: get_model() & get_transform()
│   ├── model.pth                        # Bobot Super-Ensemble terkompresi FP16 (199.46 MB)
│   └── requirements.txt                 # Dependensi inferensi evaluasi (timm)
├── notebooks/
│   ├── cnn-sampah.py                    # Skrip end-to-end pipeline Super-Ensemble Crookie
│   ├── common.py                        # Utilitas dHash, Union-Find, augmentasi, & LLRD
│   └── catatan_perbaikan_pipeline_v5_1 (2).md # Dokumentasi teknis dan log rekayasa
├── diagnostics/                         # Visualisasi training loss, confusion matrix, & error list
├── evaluate.py                          # Skrip evaluasi resmi standar asisten dosen
└── README.md                            # Laporan utama dan dokumentasi repositori
```
