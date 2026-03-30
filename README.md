# MarketSync — Marketing Channel Optimization DSS

> Decision Support System berbasis PCA + K-Means + SAW untuk optimasi channel marketing.

**Dibuat oleh:** Clarisya Adeline · Nazwa Nashatasya · Ammara Azwadiena Alfiantie  
**Mata Kuliah:** UAS Decision Support System

---

## Deskripsi

MarketSync adalah aplikasi web interaktif yang membantu tim marketing dalam mengidentifikasi segmen pelanggan dan menentukan channel pemasaran optimal menggunakan pendekatan:

- **PCA** (Principal Component Analysis) — reduksi dimensi
- **K-Means Clustering** — segmentasi pelanggan
- **SAW / MCDM** (Simple Additive Weighting) — perankingan channel

---

## Fitur Utama

| Tab | Fitur |
|-----|-------|
| **Preprocessing Pipeline** | Cleaning, feature engineering, outlier removal, normalisasi, PCA |
| **Cluster Analysis** | Elbow method, silhouette score, visualisasi PCA scatter, profil segmen |
| **SAW Decision Matrix** | Pembobotan kriteria otomatis (SDM), matriks normalisasi, skor & ranking channel |
| **Business Recommendations** | Rekomendasi strategis, analisis produk, benchmarking antar segmen |

---

## Dataset

Aplikasi menggunakan dataset **Customer Personality Analysis** dari Kaggle.

🔗 [Download dataset di sini](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

### Kolom yang Wajib Ada

| Kolom | Keterangan |
|-------|-----------|
| `Year_Birth` | Tahun lahir pelanggan |
| `Income` | Pendapatan tahunan (USD) |
| `Dt_Customer` | Tanggal bergabung |
| `NumStorePurchases` | Pembelian di toko |
| `NumWebPurchases` | Pembelian via web |
| `NumCatalogPurchases` | Pembelian via katalog |
| `MntWines` | Pengeluaran wine (2 tahun) |
| `MntFruits` | Pengeluaran buah (2 tahun) |
| `MntMeatProducts` | Pengeluaran daging (2 tahun) |
| `MntFishProducts` | Pengeluaran ikan (2 tahun) |
| `MntSweetProducts` | Pengeluaran makanan manis (2 tahun) |
| `MntGoldProds` | Pengeluaran produk emas (2 tahun) |

---

## Cara Menjalankan Secara Lokal

### 1. Clone repository

```bash
git clone https://github.com/ClarisyaA/MarketSync.git
cd MarketSync
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

---

## Deploy ke Streamlit Cloud

1. Push repository ini ke GitHub (pastikan `app.py` dan `requirements.txt` ada di root)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo → set main file: `app.py`
4. Klik **Deploy**

### Struktur File yang Diperlukan

```
marketsync-dss/
├── app.py               # File utama aplikasi
├── requirements.txt     # Dependencies Python
└── README.md            # Dokumentasi ini
```

---

## Alur Metodologi

```
Raw Data
    ↓
[1] Cleaning & Imputation     → Isi missing Income dengan median
    ↓
[2] Feature Engineering       → Age, Total_Spend, Tenure_Months
    ↓
[3] Outlier Removal           → Domain filter + Z-score |z| < 3
    ↓
[4] Log Transform + Scaler    → Log1p + StandardScaler
    ↓
[5] PCA (4D → 2D)             → PC1 + PC2
    ↓
[6] K-Means Clustering        → Segmen pelanggan
    ↓
[7] SAW / MCDM                → Ranking channel per segmen
    ↓
[8] Business Recommendations  → Strategi actionable
```

---

## Kriteria SAW

| Kode | Kriteria | Deskripsi |
|------|----------|-----------|
| K1 | Age Fit | Rata-rata usia pelanggan per channel |
| K2 | Spend Potential | Rata-rata total pengeluaran per channel |
| K3 | Loyalty | Rata-rata tenure (lama berlangganan) per channel |
| K4 | Channel Intensity | Total volume pembelian per channel |

Pembobotan menggunakan **Standard Deviation Method (SDM)** — kriteria dengan variasi lebih tinggi mendapat bobot lebih besar.

---

## Tech Stack

- [Streamlit](https://streamlit.io) — framework web app
- [Pandas](https://pandas.pydata.org) & [NumPy](https://numpy.org) — manipulasi data
- [Scikit-learn](https://scikit-learn.org) — PCA & K-Means
- [Plotly](https://plotly.com) — visualisasi interaktif
- [SciPy](https://scipy.org) — statistik (Z-score)

---

*MarketSync · UAS Decision Support System*