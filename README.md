# klasifikasi-berita-hoaks
# 🔍 Klasifikasi Berita Hoaks (Logistic Regression & IndoBERT) dengan Analisis Bias Sumber

Proyek machine learning untuk mengklasifikasikan berita berbahasa Indonesia ke dalam dua kelas, **hoaks** dan **fakta**, menggunakan **TF-IDF + Logistic Regression** dan **IndoBERT**. Proyek ini juga menganalisis **bias sumber**, yaitu kemungkinan model hanya mengenali ciri khas situs asal berita, bukan isi beritanya.

🌐 **Coba aplikasinya:** https://klasifikasi-berita-hoaks-csmfbvsxdegbvk5sm8zdru.streamlit.app/

🤗 **Model:** https://huggingface.co/marsyapn/indobert-hoaks

---

## 📂 Isi Repository

| File | Keterangan |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `requirements.txt` | Daftar library |
| `Klasifikasi_Berita_Hoaks.ipynb` | Notebook lengkap: data, cleaning, EDA, pelatihan, analisis bias, evaluasi |

Model tidak disimpan di repository ini karena ukurannya (±500 MB). Seluruh model diunduh otomatis dari Hugging Face Hub saat aplikasi dijalankan.

---

## 📊 Dataset

**Deteksi Berita Hoaks Indo - Dataset** (Kaggle)
https://www.kaggle.com/datasets/mochamadabdulazis/deteksi-berita-hoaks-indo-dataset

| Sumber | Label | Jumlah |
|---|---|---|
| TurnBackHoax.id | Hoaks | 11.816 |
| Kompas | Fakta | 4.499 |
| Antaranews | Fakta | 4.066 |
| Detik | Fakta | 3.576 |

Setelah cleaning dari teks mentah (kolom `narasi`), duplikat dan teks terlalu pendek dihapus sehingga tersisa **23.790 berita** (fakta 12.141, hoaks 11.649).

Dataset berlisensi *Research and Education Use Only*, sehingga tidak diunggah ulang ke repository ini.

> Azis, M. A., & Respati, A. A. (2026). *Deteksi Berita Hoaks Indo - Dataset* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/7385679

---

## 🧪 Hasil Model

| Model | Akurasi | F1 Hoaks | Hoaks Lolos | Fakta Salah | Uji 5 Teks |
|---|---|---|---|---|---|
| LR Versi A (penanda situs ada) | 98,44% | 0,9841 | 38 | 36 | - |
| LR Versi B (tanpa penanda situs) | 98,23% | 0,9819 | 48 | 36 | - |
| LR Versi C (tanpa jalan pintas) | 97,63% | 0,9757 | 64 | 49 | 1/5 |
| **IndoBERT Versi C** ⭐ | **99,26%** | **0,9925** | **26** | **9** | **3/5** |

- **Versi A:** cleaning biasa (normalisasi Unicode, huruf kecil, hapus URL, mention, hashtag, emoji, tanda baca).
- **Versi B:** Versi A ditambah penghapusan nama situs (kompas.com, antara, detik, turnbackhoax).
- **Versi C:** Versi B ditambah penghapusan angka, nama hari, kata "jakarta", dan kata khas pemeriksa fakta.

**Model optimal: IndoBERT** (`indobenchmark/indobert-base-p1`, *fine-tune* 2 epoch).

---

## 🔎 Temuan Analisis Bias Sumber

Akurasi sangat tinggi ternyata sebagian besar berasal dari **jalan pintas**, bukan pemahaman isi berita:

1. **Penanda situs.** "KOMPAS.com" muncul di 99,1% berita Kompas dan "(ANTARA)" di 92,5% berita Antaranews.
2. **Tahun pengumpulan data.** Kata "2026" mendorong ke fakta, "2024" mendorong ke hoaks.
3. **Gaya tulisan pemeriksa fakta.** Kata seperti "narasi", "beredar", "akun facebook", dan "arsip" berasal dari penjelasan tim TurnBackHoax, bukan dari hoaks aslinya.
4. **Gaya jurnalistik.** Kata "kata", "ujar", dan "mengatakan" menjadi penanda fakta terkuat.
5. **Panjang teks.** Berita fakta terpotong di ±145 kata, sehingga teks yang lebih panjang pasti berlabel hoaks.

Menghapus jalan pintas hanya menurunkan akurasi Logistic Regression dari 98,44% menjadi 97,63%. Pada **uji teks buatan sendiri**, Logistic Regression hanya benar 1 dari 5 teks dan IndoBERT 3 dari 5. Hoaks yang ditulis dengan gaya berita lolos pada kedua model.

---

## ⚠️ Keterbatasan

- Label ditentukan sepenuhnya oleh situs asal, sehingga model cenderung mempelajari ciri situs, bukan kebenaran isi.
- Model menilai **pola bahasa**, bukan kebenaran fakta. Hoaks yang ditulis rapi seperti berita bisa lolos.
- Uji teks buatan sendiri hanya lima contoh dan bersifat ilustrasi.

Selalu cek ulang informasi di [turnbackhoax.id](https://turnbackhoax.id) atau [cekfakta.com](https://cekfakta.com).

---

## 🛠️ Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📚 Referensi

- Azis, M. A., & Respati, A. A. (2026). *Deteksi Berita Hoaks Indo - Dataset*. Kaggle. https://doi.org/10.34740/KAGGLE/DS/7385679
- Riadi, A. T., Indriani, F., Mazdadi, M. I., Faisal, M. R., & Herteno, R. (2025). Cross-Temporal Generalization of IndoBERT for Indonesian Hoax News Classification. *JUTIF*.
- Ridho, M. Y., & Yulianti, E. (2024). From Text to Truth: Leveraging IndoBERT and Machine Learning Models for Hoax Detection in Indonesian News. *JITEKI*, 10(3), 544–555.
- IndoBERT: https://huggingface.co/indobenchmark/indobert-base-p1

---

📄 Lisensi kode: MIT
