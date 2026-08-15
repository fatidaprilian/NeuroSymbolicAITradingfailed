# Klarifikasi Metodologis: Konsistensi Arsitektur & Ruang Lingkup Revisi

**Menjawab Catatan Reviewer 3 (Poin 3 — Inkonsistensi DQN vs QR-DQN):**

### 1. Koreksi Terminologi
Pada naskah versi sebelumnya, arsitektur utama dinyatakan secara eksplisit sebagai "agen QR-DQN (Quantile Regression Deep Q-Network)" di paragraf pembuka Metode Penelitian. Setelah ditinjau ulang, kami menemukan bahwa fungsi *loss* dan arsitektur yang benar-benar diimplementasikan dan diuji adalah **Deep Q-Network (DQN) standar** (*squared-error loss / Huber loss*, bukan *quantile regression loss*). Ini adalah kesalahan penamaan pada naskah sebelumnya, bukan pada eksperimen. Kami mengoreksi seluruh istilah "QR-DQN" menjadi "DQN" secara konsisten di seluruh bagian naskah (Abstrak, Pendahuluan, Metode, Hasil, Pembahasan) agar sesuai dengan judul resmi paper:
> *"A NEURO-SYMBOLIC AI TRADING ARCHITECTURE COMBINING HYBRID LR–LSTM PREDICTION, DEEP Q-NETWORK, AND SYMBOLIC SAFETY NETS"*

---

### 2. Ruang Lingkup Revisi Tambahan
Selama proses revisi, kami juga menyempurnakan desain ruang aksi agen dari 3-aksi (*Buy/Sell/Hold*) menjadi 5-aksi diskrit $\{0: \text{SELL 100\%}, 1: \text{SELL 50\%}, 2: \text{HOLD}, 3: \text{BUY 50\%}, 4: \text{BUY 100\%}\}$, karena evaluasi ulang pada versi 3-aksi menunjukkan gejala *policy stagnation* (frekuensi transaksi sangat rendah, misalnya hanya 2–6 transaksi sepanjang periode uji 3–5 tahun) yang mengindikasikan agen gagal belajar kebijakan aktif akibat gesekan biaya transaksi. Perubahan ini berdampak pada:
1. **Pembaruan Hasil Prediksi & Trading**: Seluruh angka di Tabel 4 dan Tabel 5 diperbarui berdasarkan evaluasi 5-aksi yang aktif dan realistis.
2. **Penambahan Tabel 6 (Multi-Algorithm Benchmark)**: Menambahkan perbandingan performa komparatif terhadap *Double DQN*, *PPO*, dan *A2C*.
3. **Penambahan Uji Signifikansi Statistik**: Menyertakan uji *paired t-test* pada seluruh hasil harian. Kami secara terbuka dan jujur melaporkan bahwa perbedaan return antara strategi *Neuro-Symbolic* dan baseline tidak signifikan secara statistik ($p > 0,05$ pada ketiga aset), sehingga kontribusi utama direformulasikan sebagai kontrol risiko (*reduksi Maximum Drawdown*), bukan sebagai peningkat return yang signifikan.

---

### 3. Arsitektur Final
Agen 5-Action DQN menggunakan jaringan saraf *Multi-Layer Perceptron* (MLP) dengan fungsi aktivasi ReLU untuk mengestimasi nilai $Q(s, a)$ bagi 5 unit aksi diskrit:
$$\mathcal{A} = \{0: \text{SELL 100\%}, 1: \text{SELL 50\%}, 2: \text{HOLD}, 3: \text{BUY 50\%}, 4: \text{BUY 100\%}\}$$
