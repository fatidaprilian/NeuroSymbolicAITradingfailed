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

### 2. Ambang Batas Kalibrasi Terkondisi Rezim (Asset-Dependent Safety Net Veto Rules)

Sesuai kalibrasi pada `trading_env.py`, ambang batas *Symbolic Safety Net* dikalibrasi secara spesifik berdasarkan dinamika volatilitas dan struktur pasar per masing-masing aset:

| Aset Kripto | RSI Overbought Threshold | Normalized ATR (NATR) Threshold | Simple Moving Average (SMA) | Justifikasi Kalibrasi Aset |
| :---: | :---: | :---: | :---: | :--- |
| **BTC** | `RSI > 80` | `NATR > 15.0%` | `Price < SMA_30` | Dikalibrasi untuk aset berpikologis *large-cap* dengan fenomena *momentum trend* kuat. |
| **ETH** | `RSI > 78` | `NATR > 12.0%` | `Price < SMA_30` | Dikalibrasi untuk aset *mid-to-large cap* dengan sensitivitas volatilitas menengah. |
| **XRP** | `RSI > 82` | `NATR > 18.0%` | `Price < SMA_30` | Dikalibrasi untuk aset *altcoin* berpola volatilitas mikro ekstrim (*high-frequency spikes*). |
