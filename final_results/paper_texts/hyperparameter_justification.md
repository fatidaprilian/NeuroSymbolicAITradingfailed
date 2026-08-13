# Technical Justification & Analysis of Selected Hyperparameters

**Menjawab Catatan Reviewer 2 & Reviewer 3 (Poin 3 - Justifikasi Hyperparameter Tabel 2 & 3):**

### 1. Hyperparameter Agen Deep Q-Network (DQN)

| Hyperparameter | Nilai Terpilih | Justifikasi Teknis & Analisis Empiris |
| :--- | :---: | :--- |
| **Learning Rate ($lpha$)** | `0.0003` | Nilai *learning rate* standar yang terbukti stabil untuk jaringan MLP pada *time-series* kripto berderau tinggi. Nilai $lpha > 0.001$ menyebabkan osilasi gradien pada fungsi loss Bellman, sedangkan $lpha < 0.0001$ memperlambat konvergensi *policy*. |
| **Discount Factor ($\gamma$)** | `0.99` | Menjamin agen memperhitungkan nilai ekuitas jangka panjang (*long-term portfolio value*) hingga horison efektif $\sim 100$ jam ke depan ($rac{1}{1 - 0.99}$). |
| **Buffer Size** | `50,000` | Memori *experience replay buffer* yang optimal untuk menyimpan sampel transisi dari berbagai rezim pasar (*bullish*, *bearish*, *sideways*) tanpa menyebabkan *out-of-memory* pada RAM CPU. |
| **Batch Size** | `64` | Menyeimbangkan kestabilan estimasi gradien stokastik dan efisiensi waktu komputasi pelatihan per *epoch*. |
| **Target Update Interval** | `500` | Frekuensi pembaruan bobot *target network* untuk meminimalkan *overestimation bias* pada nilai Q-value Bellman. |
| **Exploration ($\epsilon$-decay)** | `1.0` $ightarrow$ `0.05` | Skema $\epsilon$-greedy yang menjamin eksplorasi penuh pada awal pelatihan, kemudian meluruh secara bertahap hingga menyisa 5% eksplorasi acak untuk menjaga kestabilan *policy*. |

---

### 2. Ambang Batas Symbolic Safety Net (Veto Rules)

| Parameter Safety Net | Ambang Terpilih | Justifikasi Teknis & Analisis Risiko |
| :--- | :---: | :--- |
| **Normalized ATR (NATR)** | `> 12%` | Mengidentifikasi lonjakan volatilitas mikro ekstrim pada persentil ke-95 pergerakan harga harian. Pada XRP, ambang ini secara efektif memblokir 8 transaksi beli berisiko tinggi di puncak *spike*. |
| **Relative Strength Index (RSI)** | `> 70` | Ambang batas teknis universal indikator *overbought*. Mencegah agen melakukan ekspansi posisi beli (*BUY_HALF* / *BUY_ALL*) saat harga koin berada di area jenuh beli. |
| **Simple Moving Average (SMA)** | `Price < SMA_30` | Menandai rezim *downtrend* jangka pendek (30 jam). Eksekusi beli saat harga di bawah SMA30 diblokir oleh *Symbolic Safety Net* untuk mencegah kecenderungan *catching a falling knife*. |
