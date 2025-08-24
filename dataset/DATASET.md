# Dataset Layout & Preparation

A unified folder layout for ultrasound datasets used in this project.  
**✅ `BUSBRA`, `BUSI`, `UCLM`, `UDIAT` are created *after preprocessing*.**  
**📁 `original`** contains the raw downloads **and** the preprocessing scripts.

---

## 📦 Folder Structure

```text
dataset/
├─ BUSBRA/                 # ✅ generated after preprocessing
│  ├─ full/
│  │  ├─ imgs/
│  │  └─ masks/
│  ├─ train/
│  ├─ valid/
│  ├─ full.txt             # list of full/imgs/... paths
│  ├─ train.txt            # list of train/imgs/... paths
│  └─ valid.txt            # list of valid/imgs/... paths
├─ BUSI/                   # ✅ generated after preprocessing
├─ UCLM/                   # ✅ generated after preprocessing
├─ UDIAT/                  # ✅ generated after preprocessing
├─ original/               # 📁 raw datasets + 🛠️ preprocessing scripts
│  ├─ BUS-UCLM/            # ⬇️ downloaded dataset
│  ├─ BUSBRA/              # ⬇️ downloaded dataset
│  ├─ Dataset_BUSI_with_GT/# ⬇️ downloaded dataset
│  ├─ DatasetB2/           # ⬇️ downloaded dataset
│  ├─ busbra.py            # 🛠️ processing & split
│  ├─ busi.py              # 🛠️ processing & split
│  ├─ uclm.py              # 🛠️ processing & split
│  └─ udiat.py             # 🛠️ processing & split
├─ DATASET.md
├─ preprocess.sh           # 🔁 preprocessing entrypoint   
└─ visualize.ipynb
```

## 🔗 Downloads

This work is conducted on 4 ultrasound datasets: [BUSBRA](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset), [BUSI](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset), [UCLM](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset), [UDIAT](https://www.kaggle.com/datasets/jarintasnim090/udiat-data).

📌 **After downloading**, place each dataset inside [dataset/original](https://github.com/nycu-acm/UIStyler/tree/main/dataset/original).

## ⚙️ Preprocessing

🎯 Run the preprocessing to convert downloaded datasets into the unified format
(full/, train/, valid/, each with imgs/ and masks/):
```
sh preprocess.sh
```
* ✅ Resizing policy: **BILINEAR** for images, **NEAREST** for masks.

* ✅ `train.txt`, `valid.txt`, `full.txt` list relative paths like `<DOMAIN>/<SPLIT>/imgs/<filename>`.
