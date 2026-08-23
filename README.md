# financial-forecasting-augmentation

**Mentors:** Mohammad Husain & Antoine Si | Cal Poly Pomona PolySec Lab

---
# Story 
 Every paper in this space reporting 99% accuracy is probably wrong, and I will prove it, Phase 1 reproduces those inflated results, while Part 2 diagnoses why they happened and fixes the methodology.

---

## Core Problem

if you train an LSTM on stock prices and test it with a random train/test split, you get R² > 0.99 and directional accuracy > 94%. looks incredible. the issue is that a zero-parameter persistence model (predict tomorrow = today) also gets R² > 0.95 on the same data. that's not your model being smart, that's autocorrelation inflating every metric in the benchmark.

I call this the **Autocorrelation Inflation Effect (AIE)**.

To quantify how inflated the published numbers are, i introduce the **Leakage Inflation Ratio (LIR)**:

```
LIR = Part1_metric / Part2_metric
```

across all model configurations, the average LIR for directional accuracy is ~1.84x. Part 1 reported accuracy roughly twice what a properly evaluated model achieves.

---

## Contributions

- **Autocorrelation Inflation Effect (AIE)** ┈➤ proves price-level R² is invalid as a financial ML benchmark
- **Leakage Inflation Ratio (LIR)** ┈➤ first standardized metric for quantifying evaluation inflation across studies
- **Regime-Conditional Augmentation Hypothesis (RCAH)** ┈➤ augmentation efficacy correlates with realized volatility
- **data quality primacy** ┈➤ CycleGAN augmentation (+3.7% DA) outperforms QLSTM (-42.87% DA)

---

## Results

**AIE confirmed on all 4 assets**

| asset | persistence R² | AIE confirmed |
|-------|---------------|---------------|
| AAPL | 0.9452 | yes |
| MSFT | 0.9574 | yes |
| GOOGL | 0.9455 | yes |
| BTC-USD | 0.9844 | yes (Strongest) |

**Baseline LSTM under Honest Evaluation**

| asset | DA (mean ± std) | IC | p-value |
|-------|----------------|----|---------|
| AAPL | 0.4667 ± 0.0146 | -0.014 | 0.0102 |
| MSFT | 0.4926 ± 0.0115 | -0.054 | 0.2704 |
| GOOGL | 0.4813 ± 0.0233 | -0.093 | 0.1840 |
| BTC-USD | 0.4929 ± 0.0296 | -0.012 | 0.6559 |

the ~47-49% DA is expected. it's consistent with efficient market expectations. Part 1's 94.6% was entirely from data leakage.

---

## Structure

```
financial-forecasting-augmentation/
├── notebooks/
│   ├── day1_baseline.py        # baseline LSTM x 4 assets x 5 trials
│   ├── day2_augmentation.py    # CycleGAN, WGAN-GP, SMOTE-TS (coming)
│   └── day6_qlstm_ablation.py  # quantum ablation study (coming)
├── utils/
│   ├── features.py             # RSI, SMA, Bollinger Bands, Vol
│   ├── metrics.py              # IC, return-space R², LIR, Sharpe
│   └── training.py             # LSTM model + training loop
├── results/                    # CSVs from experiments
├── figures/                    # paper figures
├── data/                       # pulled at runtime via yfinance
├── requirements.txt
└── README.md
```

---

## setup

```bash
git clone https://github.com/vnnviv/financial-forecasting-augmentation
cd financial-forecasting-augmentation
pip install -r requirements.txt
```

run on Kaggle (GPU T4) or Google Colab. yfinance pulls data automatically.

```bash
python notebooks/day1_baseline.py
```

expected runtime: ~3-4 minutes for 4 assets x 5 trials on T4.

---

## Applcations

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

PyTorch, PennyLane, yfinance, scikit-learn, Kaggle T4 GPU

---

## Paper 

Working toward arXiv submission (q-fin.ST). draft in progress.

*Diagnosing Evaluation Artifacts in Synthetic Data-Augmented Financial Forecasting: The Autocorrelation Inflation Effect and Leakage Inflation Ratio*

---

## Huge Acknowledgments
Thank you to Professor Mohammad Husain and Antoine Si at the Cal Poly Pomona PolySec Lab for mentoring this research. Presented Part 1 at SCCUR 2025.
