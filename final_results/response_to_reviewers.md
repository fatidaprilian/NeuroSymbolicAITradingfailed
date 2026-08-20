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
- **Bitcoin (BTC)**: $t = -0.155$, $p = 0.8771$
- **Ethereum (ETH)**: $t = 0.315$, $p = 0.7525$
- **Ripple (XRP)**: $t = 1.808$, $p = 0.0707$

Hasil uji statistik menunjukkan bahwa nilai $p > 0,05$ pada seluruh aset, yang berarti variansi return harian rata-rata tidak berbeda secara signifikan pada tingkat signifikansi 5%. 

Oleh karena itu, kami **mereformulasikan klaim kontribusi utama secara jujur dan ilmiah**: kontribusi *Symbolic Safety Net* difokuskan sebagai **mekanisme kontrol risiko non-intrusif dan penekan Maximum Drawdown (MDD)** (mengurangi MDD pada ETH sebesar 9,30% dan XRP sebesar 10,50%, serta memicu total 444 blokir penyelamatan modal), bukan sebagai peningkat return harian yang signifikan. Hasil ini disajikan pada **Tabel 5** dan dibahas pada **Bab 4**.

---

### Komentar 2: Justifikasi dan Analisis Hyperparameter (Tabel 2 & 3)
> *"Perlu analisis/justifikasi hyperparameter — kenapa nilai tertentu dipilih (Tabel 2 & 3), bukan sekadar daftar nilai."*

**Tanggapan Penulis:**
Kami telah menambahkan narasi justifikasi teknis dan analisis empiris untuk seluruh pemilihan hyperparameter:
1. **Learning Rate ($\alpha = 0.0003$)**: Nilai optimal untuk menjaga kestabilan konvergensi gradien loss Bellman pada *time-series* kripto berderau tinggi.
2. **Discount Factor ($\gamma = 0.99$)**: Menjamin agen memperhitungkan nilai ekuitas jangka panjang hingga horison $\sim 100$ jam ke depan.
3. **Experience Replay Buffer ($50.000$)**: Memori transisi yang cukup besar untuk memutus korelasi temporal berbagai rezim pasar.
4. **Target Network Update Interval ($500$)**: Menjaga kestabilan estimasi *Q-target* dan meminimalkan *overestimation bias*.
5. **Ambang Batas Terkondisi Distribusi Empiris (*Distribution-Calibrated Safety Net*)**:
   Sesuai literatur *Safe RL* (Zhang et al., 2024; Emam et al., 2021), ambang batas statis arbitrari rentan tidak sensitif (*under-trigger*) pada data lilin 1-jam. Oleh karena itu, ambang batas dikalibrasi berdasarkan persentil ke-90 ($Q_{0.90}$) Normalized ATR dan persentil ke-95 RSI pada data latih:
   - **BTC**: `RSI > 68.0`, `NATR > 1.23%`, `Price < SMA_30`
   - **ETH**: `RSI > 68.0`, `NATR > 1.46%`, `Price < SMA_30`
   - **XRP**: `RSI > 68.0`, `NATR > 2.26%`, `Price < SMA_30`

Penjelasan lengkap telah ditambahkan pada **Sub-bab 2.3**.

---

### Komentar 3: Analisis Tren Ekuitas Menurun pada Periode Bear Market
> *"Jelaskan penurunan kurva ekuitas portofolio selama periode pengujian."*

**Tanggapan Penulis:**
Periode pengujian (2024–2026) mencakup rezim *macro bear market* di mana harga *spot* aset mengalami kontraksi hingga -50%. Karena lingkungan perdagangan dibatasi pada pasar *spot* (tanpa *short-selling*), agen menghadapi keterbatasan struktural saat harga pasar turun berkepanjangan (Augustin et al., 2023; Omole & Enke, 2024). Namun, agen *Neuro-Symbolic* terbukti berhasil menekan kerugian dan membatasi *Maximum Drawdown* lebih baik daripada *Pure Baseline* melalui lapisan veto simbolik (444 total intervensi pada DQN). Keterbatasan pasar *spot* dan potensi perluasan ke pasar *futures/short-selling* telah ditambahkan pada **Bab 4 (Future Work & Limitations)**.

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
- **Double DQN Neuro-Symbolic** mengungguli baseline Double DQN murni sebesar **+11,16% pada BTC** (-23,69% vs -34,85%, dengan 38 blokir) dan **+9,94% pada XRP** (-40,54% vs -50,48%, dengan 105 blokir).
- Pada **ETH**, strategi *Double DQN Neuro-Symbolic* mencatatkan 214 blokir pengaman dan return -24,77%, membuktikan fleksibilitas mekanisme veto pada berbagai varian Q-learning.
- Algoritma *On-Policy* (**PPO dan A2C**) menunjukkan variansi perilaku yang tinggi di bawah gesekan biaya transaksi: PPO aktif mengeksekusi rebalancing pada ETH (294 transaksi) dan XRP (1 transaksi, 311 blokir), namun mengalami *policy collapse* pasif pada BTC baseline dan ETH A2C baseline. Perilaku ini didukung oleh temuan konsensus literatur *Safe RL* (Sebastião & Godinho, 2021; Yu et al., 2026) mengenai sensitivitas algoritma *on-policy* tanpa *experience replay* terhadap friksi biaya transaksi.
- Desain ruang aksi 5-unit diskrit terbukti efektif mengatasi *zero-trade policy stagnation* pada keluarga algoritma *off-policy* (DQN & Double DQN), mempertahankan keteraturan transaksi aktif (314–818 transaksi).

---

### Komentar 4: Abstrak Kuantitatif
> *"Lengkapi abstrak dengan hasil angka kuantitatif konkret."*

**Tanggapan Penulis:**
Abstrak telah diperbarui secara kuantitatif (dalam Bahasa Indonesia dan Bahasa Inggris) dengan mencantumkan: 818 transaksi BTC (119 blokir), 314 transaksi ETH (114 blokir), 572 transaksi XRP (211 blokir), total 444 blokir penyelamatan modal, peningkatan return +10,80% pada XRP (-33,39% vs -44,19%), dan reduksi MDD dari -40,90% menjadi -31,60% pada ETH dan dari -45,58% menjadi -35,08% pada XRP.

---

### Komentar 5: Tabel Penelitian Terdahulu (Related Work)
> *"Tambahkan tabel pemetaan penelitian terdahulu yang komprehensif."*

**Tanggapan Penulis:**
Kami telah menyertakan **Tabel 1 (Comprehensive Related Work Table 2021–2026)** di Bagian Pendahuluan, memetakan 10 penelitian terkini (termasuk Kabbani & Duman 2022, Kochliaridis et al. 2023, Muminov et al. 2024, Huang & Su 2024, Vergara & Kristjanpoller 2024, Otabek & Choi 2024, Zhang 2025, Priya et al. 2025, Khujamatov et al. 2026, dan Jiang et al. 2026) dengan DOI resmi.
