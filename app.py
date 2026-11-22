import re
import unicodedata

import joblib
import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_MODEL = "marsyapn/indobert-hoaks"

st.set_page_config(page_title="Cek Hoaks", page_icon="🔍", layout="centered")


# ---------------------------------------------------------------
# Cleaning (harus sama persis dengan notebook, Versi C)
# ---------------------------------------------------------------
POLA_PENANDA = r"kompas\.com|\(antara\)|antaranews|detik\w*|turnbackhoax"

KATA_PINTAS = [
    "jakarta", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
    "narasi", "beredar", "akun", "facebook", "arsip", "dibagikan", "membagikan",
    "unggahan", "mengunggah", "diunggah", "benarkah", "klaim",
]
POLA_PINTAS = r"\b(" + "|".join(KATA_PINTAS) + r")\b"


def hapus_penanda(teks):
    return re.sub(POLA_PENANDA, " ", str(teks), flags=re.IGNORECASE)


def bersihkan(teks):
    t = unicodedata.normalize("NFKC", str(teks))
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def hapus_pintas(teks):
    t = re.sub(r"\d+", " ", teks)
    t = re.sub(POLA_PINTAS, " ", t)
    return re.sub(r"\s+", " ", t).strip()


def siapkan_teks(teks):
    return hapus_pintas(bersihkan(hapus_penanda(teks)))


# ---------------------------------------------------------------
# Memuat model dari Hugging Face Hub (sekali saja, lalu disimpan)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model, mohon tunggu sebentar...")
def muat_model():
    tokenizer = AutoTokenizer.from_pretrained(REPO_MODEL)
    model_bert = AutoModelForSequenceClassification.from_pretrained(REPO_MODEL)
    model_bert.eval()
    model_lr = joblib.load(hf_hub_download(REPO_MODEL, "model_lr_hoaks.pkl"))
    tfidf = joblib.load(hf_hub_download(REPO_MODEL, "tfidf_hoaks.pkl"))
    return tokenizer, model_bert, model_lr, tfidf


def prediksi_bert(teks_bersih, tokenizer, model_bert):
    enc = tokenizer(teks_bersih, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        prob = torch.softmax(model_bert(**enc).logits, dim=1)[0][1].item()
    return prob


def prediksi_lr(teks_bersih, model_lr, tfidf, n_kata=8):
    x = tfidf.transform([teks_bersih])
    prob = model_lr.predict_proba(x)[0][1]

    # Kontribusi tiap kata = nilai TF-IDF x bobot Logistic Regression
    kata = tfidf.get_feature_names_out()
    idx = x.nonzero()[1]
    kontribusi = x.toarray()[0][idx] * model_lr.coef_[0][idx]
    urut = np.argsort(kontribusi)
    kata_hoaks = [(kata[idx[i]], kontribusi[i]) for i in urut[::-1][:n_kata] if kontribusi[i] > 0]
    kata_fakta = [(kata[idx[i]], kontribusi[i]) for i in urut[:n_kata] if kontribusi[i] < 0]
    return prob, kata_hoaks, kata_fakta


def label(prob):
    return ("HOAKS", "🔴") if prob >= 0.5 else ("FAKTA", "🟢")


# ---------------------------------------------------------------
# Tampilan
# ---------------------------------------------------------------
st.title("🔍 Cek Hoaks")
st.write(
    "Tempel teks berita atau pesan berantai, lalu periksa apakah polanya "
    "mirip **hoaks** atau **berita fakta**."
)

CONTOH = {
    "Pilih contoh...": "",
    "Pesan bantuan tunai (gaya medsos)": (
        "VIRAL!! Bagi yang punya kartu ATM silahkan klik link ini untuk dapat bantuan tunai "
        "2 juta dari pemerintah, cepat sebelum kuota habis, sebarkan ke semua grup!!!"
    ),
    "Info cuaca BMKG (gaya berita)": (
        "Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) memperkirakan sebagian wilayah "
        "Indonesia akan mengalami hujan dengan intensitas sedang hingga lebat dalam sepekan ke "
        "depan. Masyarakat diimbau tetap waspada terhadap potensi banjir dan tanah longsor, "
        "ujar kepala BMKG."
    ),
    "Klaim kesehatan (hoaks bergaya berita)": (
        "Kementerian Kesehatan mengatakan bahwa minum air kelapa yang dicampur garam setiap pagi "
        "dapat menyembuhkan diabetes dalam waktu tiga hari. Hal tersebut disampaikan dalam "
        "konferensi pers. Masyarakat diimbau untuk segera mencobanya, ujar juru bicara kementerian."
    ),
}

pilihan = st.selectbox("Atau coba salah satu contoh:", list(CONTOH.keys()))
teks = st.text_area("Teks berita / pesan", value=CONTOH[pilihan], height=180,
                    placeholder="Tempel teks di sini...")

if st.button("Periksa", type="primary"):
    teks_bersih = siapkan_teks(teks)

    if len(teks_bersih.split()) < 5:
        st.warning("Teksnya terlalu pendek. Tempel minimal satu kalimat lengkap (5 kata atau lebih).")
        st.stop()

    tokenizer, model_bert, model_lr, tfidf = muat_model()
    prob_bert = prediksi_bert(teks_bersih, tokenizer, model_bert)
    prob_lr, kata_hoaks, kata_fakta = prediksi_lr(teks_bersih, model_lr, tfidf)

    st.subheader("Hasil")
    kol1, kol2 = st.columns(2)
    with kol1:
        lab, ikon = label(prob_bert)
        st.metric("🤖 IndoBERT (model utama)", f"{ikon} {lab}")
        st.caption(f"Peluang hoaks: {prob_bert*100:.1f}%")
        st.progress(prob_bert)
    with kol2:
        lab, ikon = label(prob_lr)
        st.metric("📊 Logistic Regression", f"{ikon} {lab}")
        st.caption(f"Peluang hoaks: {prob_lr*100:.1f}%")
        st.progress(float(prob_lr))

    if label(prob_bert)[0] != label(prob_lr)[0]:
        st.info(
            "Kedua model **tidak sepakat**. Artinya teks ini tidak memiliki pola yang jelas. "
            "Sebaiknya periksa kebenarannya secara manual."
        )

    st.subheader("Kata yang paling memengaruhi")
    st.caption("Dihitung dari model Logistic Regression.")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Mendorong ke arah hoaks**")
        if kata_hoaks:
            for k, _ in kata_hoaks:
                st.markdown(f"- 🔴 {k}")
        else:
            st.write("-")
    with k2:
        st.markdown("**Mendorong ke arah fakta**")
        if kata_fakta:
            for k, _ in kata_fakta:
                st.markdown(f"- 🟢 {k}")
        else:
            st.write("-")

    st.warning(
        "⚠️ **Penting:** model ini menilai **pola bahasa**, bukan kebenaran fakta. "
        "Hoaks yang ditulis rapi seperti berita bisa lolos, dan informasi benar yang "
        "mengandung kata seperti \"gratis\" bisa dianggap hoaks. Selalu cek ulang di "
        "[turnbackhoax.id](https://turnbackhoax.id) atau [cekfakta.com](https://cekfakta.com)."
    )

with st.expander("ℹ️ Tentang aplikasi ini"):
    st.markdown(
        """
Aplikasi ini adalah bagian dari proyek **Klasifikasi Berita Hoaks (Logistic Regression & IndoBERT)
dengan Analisis Bias Sumber**.

- **Dataset:** Deteksi Berita Hoaks Indo (Azis & Respati, 2026), berisi berita hoaks dari
  TurnBackHoax.id serta berita fakta dari Kompas, Antaranews, dan Detik.
- **Model:** IndoBERT (akurasi 99,26%) dan TF-IDF + Logistic Regression (akurasi 97,63%) pada data uji.
- **Temuan utama:** akurasi tinggi sebagian besar berasal dari perbedaan gaya penulisan antar situs
  sumber. Pada uji teks buatan sendiri, IndoBERT hanya benar 3 dari 5 dan Logistic Regression 1 dari 5.
- **Model:** [huggingface.co/marsyapn/indobert-hoaks](https://huggingface.co/marsyapn/indobert-hoaks)
"""
    )
