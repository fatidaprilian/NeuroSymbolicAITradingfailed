# Dokumen Tanggapan Resmi Penulis terhadap Reviewer (Response to Reviewers)

**Judul Paper**: *A Neuro-Symbolic AI Trading Architecture Combining Hybrid LR–LSTM Prediction, Deep Q-Network, and Symbolic Safety Nets*

Kami mengucapkan terima kasih yang sebesar-besarnya kepada Dewan Redaksi dan para Mitra Bestari (Reviewer 1, 2, dan 3) atas saran, kritik konstruktif, dan masukan berharga yang telah diberikan. Seluruh masukan telah kami tindaklanjuti secara menyeluruh dengan perbaikan naskah, penambahan data empiris, dan eksperimen komparatif.

Berikut adalah tanggapan poin-demi-poin kami terhadap komentar para Mitra Bestari:

---

## Tanggapan terhadap Mitra Bestari 1

### Komentar 1: Penulisan Istilah Bahasa Asing (Italic)
> *"Periksa kembali penulisan istilah bahasa asing di seluruh naskah agar konsisten menggunakan huruf miring (italic)."*

**Tanggapan Penulis:**
Kami telah memeriksa seluruh naskah dan memformat seluruh istilah asing (seperti *bear market*, *reward hacking*, *safety net*, *equity curve*, *drawdown*, *on-policy*, *off-policy*, *policy stagnation*, *overbought*, *veto mechanism*) menggunakan huruf miring (*italic*) secara konsisten di seluruh bagian naskah.

---

### Komentar 2: Kualitas Gambar Grafik dan Keterbacaan Sumbu
> *"Pastikan grafik hasil memiliki resolusi tinggi dan label sumbu serta tanggal tidak saling bertumpukan."*

**Tanggapan Penulis:**
Seluruh grafik kurva ekuitas (*equity curves*) dan riwayat eksekusi transaksi telah diekspor ulang dengan resolusi tinggi **200 DPI** per masing-masing subfolder aset (`final_results/charts/btc/`, `eth/`, `xrp/`). Format penanggalan pada sumbu horizontal telah dimiringkan sebesar 30 derajat dengan *padding* judul dan tata letak *legend* yang optimal sehingga bebas dari tumpang-tindih (*zero overlap*).

---

## Tanggapan terhadap Mitra Bestari 2

### Komentar 1: Uji Signifikansi Statistik (Paired t-Test)
> *"Reviewer meminta uji statistik untuk membuktikan apakah perbedaan performa baseline vs Neuro-Simbolik signifikan secara statistik (bukan hanya perbandingan angka mentah di Tabel 5)."*

**Tanggapan Penulis:**
Kami telah melakukan uji signifikansi statistik inferensial (*Paired Sample t-Test*) terhadap return harian antara strategi *Neuro-Symbolic 5-Action DQN* dan *Pure Baseline DQN* sepanjang periode uji data historis (2021–2026):
- **Bitcoin (BTC)**: $t = 0.2946$, $p = 0.7684$
- **Ethereum (ETH)**: $t = 0.2863$, $p = 0.7748$
- **Ripple (XRP)**: $t = 1.3857$, $p = 0.1665$

Hasil uji statistik menunjukkan bahwa nilai $p > 0,05$ pada seluruh aset, yang berarti variansi return harian rata-rata tidak berbeda secara signifikan pada tingkat signifikansi 5%. 

Oleh karena itu, kami **mereformulasikan klaim kontribusi utama secara jujur dan ilmiah**: kontribusi *Symbolic Safety Net* difokuskan sebagai **mekanisme kontrol risiko non-intrusif dan penekan Maximum Drawdown (MDD)** (mengurangi MDD pada ETH sebesar 8,17% dan XRP sebesar 7,70%), bukan sebagai peningkat return harian yang signifikan. Hasil ini disajikan pada **Tabel 5** dan dibahas pada **Bab 4**.

---

### Komentar 2: Justifikasi dan Analisis Hyperparameter (Tabel 2 & 3)
> *"Perlu analisis/justifikasi hyperparameter — kenapa nilai tertentu dipilih (Tabel 2 & 3), bukan sekadar daftar nilai."*

**Tanggapan Penulis:**
Kami telah menambahkan narasi justifikasi teknis dan analisis empiris untuk seluruh pemilihan hyperparameter:
1. **Learning Rate ($\alpha = 0.0003$)**: Nilai optimal untuk menjaga kestabilan konvergensi gradien loss Bellman pada *time-series* kripto berderau tinggi.
2. **Discount Factor ($\gamma = 0.99$)**: Menjamin agen memperhitungkan nilai ekuitas jangka panjang hingga horison $\sim 100$ jam ke depan.
3. **Experience Replay Buffer ($50.000$)**: Memori transisi yang cukup besar untuk memutus korelasi temporal berbagai rezim pasar.
4. **Target Network Update Interval ($500$)**: Menjaga kestabilan estimasi *Q-target* dan meminimalkan *overestimation bias*.
5. **Ambang Batas Terkondisi Rezim Per Aset (*Asset-Dependent Safety Net*)**:
   - **BTC**: `RSI > 80`, `NATR > 15.0%`, `SMA_30` (Dikalibrasi untuk aset *large-cap* dengan *momentum trend* kuat).
   - **ETH**: `RSI > 78`, `NATR > 12.0%`, `SMA_30` (Dikalibrasi untuk volatilitas menengah).
   - **XRP**: `RSI > 82`, `NATR > 18.0%`, `SMA_30` (Dikalibrasi untuk volatilitas mikro tinggi guna menghindari veto palsu).

Penjelasan lengkap telah ditambahkan pada **Sub-bab 2.3**.

---

### Komentar 3: Analisis Tren Ekuitas Menurun pada Periode Bear Market
> *"Jelaskan penurunan kurva ekuitas portofolio selama periode pengujian."*

**Tanggapan Penulis:**
Periode pengujian (2024–2026) mencakup rezim *macro bear market* di mana harga *spot* aset mengalami kontraksi hingga -50%. Karena lingkungan perdagangan dibatasi pada pasar *spot* (tanpa *short-selling*), agen menghadapi keterbatasan struktural saat harga pasar turun berkepanjangan. Namun, agen *Neuro-Symbolic* terbukti berhasil menekan kerugian dan membatasi *Maximum Drawdown* lebih baik daripada *Pure Baseline* melalui lapisan veto simbolik. Keterbatasan pasar *spot* dan potensi perluasan ke pasar *futures/short-selling* telah ditambahkan pada **Bab 4 (Future Work & Limitations)**.

---

## Tanggapan terhadap Mitra Bestari 3

### Komentar 1: Konsistensi Arsitektur DQN vs QR-DQN
> *"Perjelas konsistensi penggunaan DQN vs QR-DQN — di abstrak & pendahuluan disebut 'DQN', tapi di bagian Metode (2.1) tiba-tiba disebut 'agen QR-DQN'."*

**Tanggapan Penulis:**
Kami mengakui bahwa penyebutan "QR-DQN" pada naskah versi terdahulu adalah kesalahan penulisan terminologi pada draf awal. Fungsi *loss* dan model yang benar-benar dirancang, dilatih, dan diuji secara empiris adalah **Deep Q-Network (DQN) standar** berbasis *squared-error / Huber loss* dengan *Experience Replay Buffer* dan *Target Network*.

Kami telah **menyeragamkan 100% seluruh istilah menjadi "5-Action Deep Q-Network (DQN)"** di seluruh Abstrak, Pendahuluan, Metode, Hasil, dan Pembahasan sesuai dengan Judul Resmi Paper.

---

### Komentar 2: Metrik Kuantitatif Model Prediksi (LR–LSTM)
> *"Reviewer 3 meminta metrik performa model prediksi ditambahkan: MAE, RMSE, atau MAPE untuk LR–LSTM."*

**Tanggapan Penulis:**
Kami telah menambahkan **Tabel 4** yang menyajikan metrik performa model prediksi harga hibrida LR–LSTM hasil evaluasi pada data uji (*test set* 2024–2026):

| Aset Kripto | Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Bitcoin (BTC)** | Hybrid LR-LSTM (Usulan) | 178.41 | 272.17 | 0.26% | 0.9985 |
| **Ethereum (ETH)** | Hybrid LR-LSTM (Usulan) | 6.45 | 9.97 | 0.34% | 0.9982 |
| **Ripple (XRP)** | Hybrid LR-LSTM (Usulan) | 0.004 | 0.007 | 0.36% | 0.9981 |

Data ini dimasukkan pada **Sub-bab 3.1**.

---

### Komentar 3: Perbandingan dengan Algoritma RL Lain (Double DQN, PPO, A2C)
> *"Perlu perbandingan dengan algoritma reinforcement learning lain seperti PPO, SAC, atau Double DQN untuk memperkuat validitas hasil penelitian."*

**Tanggapan Penulis:**
Kami telah melatih dan menguji secara komparatif algoritma **Double DQN, PPO, dan A2C** pada ketiga aset kripto (BTC, ETH, XRP). Hasil komparatif disajikan pada **Tabel 6**:
- **Double DQN Neuro-Symbolic** mengungguli baseline Double DQN murni sebesar **+12,11% pada BTC** (-22,74% vs -34,85%) dan **+10,56% pada XRP** (-39,92% vs -50,48%).
- Namun demikian, untuk kombinasi **Double DQN pada ETH**, strategi *Pure Baseline* mencatat performa yang lebih baik dibandingkan *Neuro-Symbolic* (-17,74% vs -23,62%), yang mengindikasikan bahwa efektivitas *Safety Net* dapat bervariasi tergantung interaksi antara arsitektur estimasi fungsi nilai (*Q-value*) dan profil momentum aset. Temuan ini kami laporkan secara terbuka dan dibahas pada Bab 4 sebagai dasar perlunya kalibrasi ambang batas adaptif per-algoritma untuk penelitian selanjutnya.
- Algoritma *On-Policy* (**PPO dan A2C**) mengalami fenomena *Action Stagnation / Local Minimum Collapse* (0 transaksi, 0.00% return) karena tidak memiliki *Experience Replay Buffer* untuk memutus korelasi temporal dalam ruang *trading* dengan biaya transaksi. Fenomena ini dianalisis dan didukung oleh literatur *policy-making RL* (arXiv:2312.06527).
- Desain ruang aksi 5-unit diskrit dirancang untuk mengatasi fenomena *zero-trade policy stagnation* yang rentan terjadi pada ruang aksi 3-unit standar (*buy/hold/sell*) di bawah gesekan biaya transaksi (Vergara & Kristjanpoller, 2024; Kumlungmak, 2022; Huang & Su, 2024).

---

### Komentar 4: Abstrak Kuantitatif
> *"Lengkapi abstrak dengan hasil angka kuantitatif konkret."*

**Tanggapan Penulis:**
Abstrak telah diperbarui secara kuantitatif (dalam Bahasa Indonesia dan Bahasa Inggris) dengan mencantumkan: 830 transaksi BTC, 324 transaksi ETH, 614 transaksi XRP, peningkatan return +7,94% pada XRP, reduksi MDD 8,17% pada ETH, dan 8 blokir veto simbolik pada XRP.

---

### Komentar 5: Tabel Penelitian Terdahulu (Related Work)
> *"Tambahkan tabel pemetaan penelitian terdahulu yang komprehensif."*

**Tanggapan Penulis:**
Kami telah menyertakan **Tabel 1 (Comprehensive Related Work Table 2021–2026)** di Bagian Pendahuluan, memetakan 10 penelitian terkini (termasuk Kabbani & Duman 2022, Kochliaridis et al. 2023, Muminov et al. 2024, Huang & Su 2024, Vergara & Kristjanpoller 2024, Otabek & Choi 2024, Zhang 2025, Priya et al. 2025, Khujamatov et al. 2026, dan Jiang et al. 2026) dengan DOI resmi.
