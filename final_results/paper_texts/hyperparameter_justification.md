# Technical Justification & Analysis of Selected Hyperparameters

**Menjawab Catatan Reviewer 2 & Reviewer 3 (Poin 3 - Justifikasi Hyperparameter Tabel 2 & 3):**

### 1. Hyperparameter Agen Deep Q-Network (DQN)

| Hyperparameter | Nilai Terpilih | Justifikasi Teknis & Analisis Empiris |
| :--- | :---: | :--- |
| **Learning Rate ($\alpha$)** | `0.0003` | Nilai *learning rate* standar yang terbukti stabil untuk jaringan MLP pada *time-series* kripto berderau tinggi. Nilai $\alpha > 0.001$ menyebabkan osilasi gradien pada fungsi loss Bellman, sedangkan $\alpha < 0.0001$ memperlambat konvergensi *policy*. |
| **Discount Factor ($\gamma$)** | `0.99` | Menjamin agen memperhitungkan nilai ekuitas jangka panjang (*long-term portfolio value*) hingga horison efektif $\sim 100$ jam ke depan ($\frac{1}{1 - 0.99}$). |
| **Buffer Size** | `50,000` | Memori *experience replay buffer* yang optimal untuk menyimpan sampel transisi dari berbagai rezim pasar (*bullish*, *bearish*, *sideways*) tanpa menyebabkan *out-of-memory* pada RAM CPU. |
| **Batch Size** | `64` | Menyeimbangkan kestabilan estimasi gradien stokastik dan efisiensi waktu komputasi pelatihan per *epoch*. |
| **Target Update Interval** | `500` | Frekuensi pembaruan bobot *target network* untuk meminimalkan *overestimation bias* pada nilai Q-value Bellman. |
| **Exploration ($\epsilon$-decay)** | `1.0` $\rightarrow$ `0.05` | Skema $\epsilon$-greedy yang menjamin eksplorasi penuh pada awal pelatihan, kemudian meluruh secara bertahap hingga menyisa 5% eksplorasi acak untuk menjaga kestabilan *policy*. |

---

### 2. Ambang Batas Kalibrasi Terkondisi Distribusi (Distribution-Calibrated Safety Net Veto Rules)

Sesuai konsensus Safe Reinforcement Learning (Zhang et al., 2024; Emam et al., 2021; Su et al., 2024), ambang batas statis arbitrari rentan mengalami *under-triggering* saat pergeseran rezim pasar. Oleh karena itu, ambang batas *Symbolic Safety Net* pada `trading_env.py` dikalibrasi berdasarkan persentil ke-90 ($Q_{0.90}$) distribusi empiris Normalized ATR (NATR) dan persentil ke-95 RSI pada data lilin 1-jam:

| Aset Kripto | RSI Overbought Threshold | Normalized ATR (NATR) Threshold | Simple Moving Average (SMA) | Justifikasi Kalibrasi Distribusi Empiris |
| :---: | :---: | :---: | :---: | :--- |
| **BTC** | `RSI > 68.0` | `NATR > 1.23%` | `Price < SMA_30` | Dikalibrasi pada persentil ke-90 NATR lilin 1-jam BTC ($Q_{0.90} = 1.23\%$) untuk mendeteksi anomali volatilitas aset *large-cap*. |
| **ETH** | `RSI > 68.0` | `NATR > 1.46%` | `Price < SMA_30` | Dikalibrasi pada persentil ke-90 NATR lilin 1-jam ETH ($Q_{0.90} = 1.46\%$) untuk sensitivitas volatilitas menengah. |
| **XRP** | `RSI > 68.0` | `NATR > 2.26%` | `Price < SMA_30` | Dikalibrasi pada persentil ke-90 NATR lilin 1-jam XRP ($Q_{0.90} = 2.26\%$) guna memitigasi *micro-volatility spikes* altcoin. |
