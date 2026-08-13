# Methodological Clarification: DQN Architecture Consistency

**Menjawab Catatan Reviewer 3 (Poin 3 - Inkonsistensi DQN vs QR-DQN):**

### 1. Definisi & Konsistensi Model Utama
- Model utama yang dirancang, dilatih, dan diuji secara riil dalam repositori ini adalah **5-Action Deep Q-Network (DQN)** berbasis *Value-based Q-learning* standar dengan *Experience Replay Buffer* dan *Target Network*.
- Pada draf versi terdahulu, istilah *QR-DQN (Quantile Regression Deep Q-Network)* sempat tersebut pada latar belakang perbandingan *Distributional RL*.
- Untuk menjamin **konsistensi metodologis 100%** di seluruh bagian naskah paper (Abstrak, Pendahuluan, Metode, Hasil Eksperimen, dan Pembahasan), nama arsitektur yang diusulkan **DISERAGAMKAN 100% MENJADI: "5-Action Deep Q-Network (DQN)"**, sesuai dengan Judul Resmi Paper:
  > *"A NEURO-SYMBOLIC AI TRADING ARCHITECTURE COMBINING HYBRID LR–LSTM PREDICTION, DEEP Q-NETWORK, AND SYMBOLIC SAFETY NETS"*

---

### 2. Hubungan DQN dan Ruang Aset 5-Aksi
- Agen **5-Action DQN** menggunakan jaringan saraf *Multi-Layer Perceptron (MLP)* dengan fungsi aktivasi ReLU untuk mengestimasi nilai *Q-value* $Q(s, a)$ bagi 5 unit aksi diskrit:
  $$\mathcal{A} = \{0: \text{SELL 100\%}, 1: \text{SELL 50\%}, 2: \text{HOLD}, 3: \text{BUY 50\%}, 4: \text{BUY 100\%}\}$$
- Penggunaan ruang aksi 5-unit ini secara eksplisit memecahkan masalah *zero-trade policy stagnation* pada 3-aksi standar (*all-in BUY/SELL*) seperti yang didokumentasikan oleh Kaur et al. (2025).
