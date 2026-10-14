# Studio 2 — Klasifikasi Sampah · Kelompok <1>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No  | Nama Lengkap            | NIM       | Kontribusi utama |
| --- | ----------------------- | --------- | ---------------- |
| 1   | George Haansraj         | 123140001 |                  |
| 2   | Mario Fransiskus Sitepu | 123140023 |                  |
| 3   |                         |           |                  |

## Ringkasan Pendekatan

Proyek ini menggunakan **ConvNeXt-Tiny pretrained ImageNet-1K** dengan pendekatan transfer learning dan fine-tuning untuk mengklasifikasikan gambar sampah ke dalam lima kelas:

1. `botol_plastik`
2. `gelas_plastik`
3. `kertas_kardus`
4. `saset_kemasan`
5. `styrofoam`

- **Arsitektur / backbone**: ConvNeXt-Tiny (pretrained ImageNet-1K)
- **Strategi pelatihan**: Model dilatih menggunakan data training yang telah diaudit dan dibagi menjadi training set serta validation set. Strategi pelatihan menggunakan tahap warmup pada classification head, dilanjutkan dengan fine-tuning seluruh model.
- **Data tambahan**: Tidak ada (hanya menggunakan dataset yang disediakan untuk kompetisi).
- **Ukuran input & preprocessing**: 224x224 (Resize 236 bilinear + CenterCrop 224 + Normalisasi ImageNet)

## Pretrained Model dan Sumber Data

- **Arsitektur pretrained:** ConvNeXt-Tiny, ImageNet-1K.
- **Dataset:** Dataset yang disediakan untuk kompetisi.
- **Data tambahan:** Tidak ada.

## Hasil

| Putaran            | Commit    | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
| ------------------ | --------- | ------------------------------- | ---------------------------- |
| Preview 1 (14 Okt) | `50e8a37` | 97,83%                          |                              |
| Preview 2 (21 Okt) |           |                                 |                              |
| Final (25 Okt)     |           |                                 |                              |

### Hasil Validasi Lokal

| Metrik | Hasil |
|---|---:|
| Best Validation Macro-F1 | 97,83% |
| Validation Accuracy | 97,82% |
| Jumlah gambar validasi | 551 |

*Catatan: hasil di atas merupakan hasil validasi lokal, bukan hasil hidden validation set kompetisi.*

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- Eksperimen backbone ConvNeXt-Tiny menunjukkan konvergensi dan separasi fitur yang sangat baik pada klasifikasi 5 kelas sampah.
- Penyesuaian preprocessing (Resize 236 -> CenterCrop 224) menjaga konsistensi aspect ratio dan fitur gambar.

## Cara Reproduksi Evaluasi

Pastikan Python dan dependensi yang diperlukan telah terpasang, serta data validasi tersedia dalam struktur folder sesuai nama kelas yang ditetapkan.

```bash
python evaluate.py --submission ./submission --data ./my_val
```

Ganti `./my_val` dengan direktori validation set lokal.

## Struktur Repositori

```
submission/
  model.py          # mendefinisikan get_model() dan get_transform() sesuai kontrak evaluator
  model.pth         # bobot model hasil fine-tuning, disimpan menggunakan Git LFS
evaluate.py         # salinan skrip evaluasi resmi
```

Catatan berkas:
- `submission/model.py`: mendefinisikan `get_model()` dan `get_transform()` sesuai kontrak evaluator.
- `submission/model.pth`: bobot model hasil fine-tuning, disimpan menggunakan Git LFS.

## Teknologi

- Python
- PyTorch
- Torchvision
- Pillow
- Git LFS
