# Dataset

## About the original "Malayalam Char Gabor" dataset

The source paper uses a proprietary, pre-extracted Gabor-feature
dataset ("Malayalam Char Gabor") with 50 numeric features + 1 label
column across 48 character classes. That exact feature file has not
been publicly released by the authors, so it isn't linked here.

Instead, this project gives you two ways to get running:

1. **Extract the features yourself** from an open Malayalam character
   image dataset using `src/gabor_feature_extraction.py` (recommended
   — this reproduces the paper's pipeline end-to-end).
2. **Bring your own CSV** of Gabor (or other numeric) features with a
   label column, and point `malayalam_char_recognition.py` directly at
   it with `--data-path` / `--label-col`.

## Open, public Malayalam character image sources

Any of these can be fed into `gabor_feature_extraction.py` (after
arranging them into one sub-folder per class, see below):

| Dataset | Source | Notes |
|---|---|---|
| Amrita_MalCharDb | https://tc11.cvc.uab.es/datasets/Amrita_MalCharDb_1 | Handwritten Malayalam character image DB from the same institution as this paper; images + CSV of 32x32 pixel vectors with class labels |
| Malayalam Handwritten Character Dataset | https://www.kaggle.com/datasets/ajayjames/malayalam-handwritten-character-dataset | Kaggle, character images by class |
| Malayalam Characters Dataset (126 characters) | https://www.kaggle.com/datasets/adarshmohan/malayalam-characters-dataset-126 | Kaggle, larger character set |
| Handwritten Malayalam | https://www.kaggle.com/datasets/amaljossy/handwritten-malayalam | Kaggle, digitally written characters |
| HMPLMD (Handwritten Malayalam Palm Leaf Manuscript Dataset) | https://www.sciencedirect.com/science/article/pii/S2352340923000781 | Actual palm-leaf manuscript images (Kambaramayanam, Jathakas) with ground-truth binarization — closest in spirit to the original palm-leaf use case, released as a Data-in-Brief article |

## Expected folder layout for feature extraction

```
data/images/
    character_01/
        img_0001.png
        img_0002.png
        ...
    character_02/
        ...
    ...
    character_48/
        ...
```

Then run:

```bash
python src/gabor_feature_extraction.py \
    --images-dir data/images \
    --output-csv data/malayalam_char_gabor.csv
```

This produces a 50-feature + `label` CSV (5 Gabor frequencies x 5
orientations x [mean, variance] per filtered image), ready to feed
into `malayalam_char_recognition.py`.

## Getting data via the Kaggle CLI

```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d ajayjames/malayalam-handwritten-character-dataset -p data --unzip
```
