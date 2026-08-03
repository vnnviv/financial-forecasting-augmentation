# Git Commands — Push to GitHub

## Step 1: Create the repo on GitHub

Go to github.com → click "New repository"
- Name: `financial-forecasting-augmentation`
- Description: `Diagnosing evaluation artifacts in synthetic data-augmented financial forecasting`
- Set to Public
- Do NOT initialize with README (we already have one)
- Click "Create repository"

---

## Step 2: Copy these commands into your terminal

```bash
# go into the repo folder you downloaded
cd vivian-financial-forecasting

# initialize git
git init

# set your name and email (first time only)
git config --global user.name "vnnviv"
git config --global user.email "your-email@gmail.com"

# connect to GitHub (replace YOUR_USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/financial-forecasting-augmentation.git

# add all files
git add .

# first commit
git commit -m "initial commit: Day 1 baseline LSTM experiments

- Baseline LSTM x 4 assets x 5 trials
- AIE proof (persistence baseline R² > 0.95)
- LIR table (mean 1.84x inflation)
- 3 figures saved to /figures
- Results saved to /results"

# push to GitHub
git push -u origin main
```

---

## Step 3: Verify it worked

Go to: `https://github.com/YOUR_USERNAME/financial-forecasting-augmentation`

You should see all your files.

---

## Future commits (after each experiment day)

```bash
# Day 2 — after running augmentation experiments
git add .
git commit -m "Day 2: CycleGAN augmentation results"
git push

# Day 6 — after QLSTM ablation
git add .
git commit -m "Day 6: QLSTM ablation study"
git push
```

---

## If you get an authentication error

GitHub stopped allowing passwords in 2021. You need a Personal Access Token:

1. Go to github.com → Settings → Developer Settings
2. Personal Access Tokens → Tokens (classic)
3. Generate new token → check "repo" scope
4. Copy the token
5. When git asks for password, paste the token instead

---

## Folder structure after pushing

```
financial-forecasting-augmentation/
├── notebooks/
│   └── day1_baseline.py
├── utils/
│   ├── __init__.py
│   ├── features.py
│   ├── metrics.py
│   └── training.py
├── results/
│   ├── day1_baseline.csv
│   └── day1_lir.csv
├── figures/
│   ├── fig1_aie.png
│   ├── fig2_lir.png
│   └── fig3_cross_asset.png
├── data/           (empty — data pulled at runtime via yfinance)
├── .gitignore
├── requirements.txt
└── README.md
```
