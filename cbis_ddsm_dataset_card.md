# Dataset Card: CBIS-DDSM

## Overview

CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is a curated subset of the Digital Database for Screening Mammography (DDSM). It was created to address limitations in the original DDSM by providing a standardized and more accessible collection of mammography images. Unlike the original DDSM which contained both positive and negative cases, CBIS-DDSM focuses specifically on abnormal cases — including both benign and malignant findings — and provides pixel-level segmentation masks that outline the lesion regions of interest. These masks make it particularly valuable for segmentation and lesion detection tasks. The dataset contains images in DICOM format, which have been converted to PNG for use in this project. It is one of the smaller datasets in our pipeline but provides high-quality ROI annotations that are well-suited for guiding the segmentation component of the model.

---

## Dataset Details

| Field | Details |
|---|---|
| Access | [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) |
| Number of Patients | 2,620 |
| Number of Images | 10,239 |
| Image Format | DICOM (converted to PNG for this project) |
| Annotations | Pixel-level segmentation masks for lesion ROI |
| Task Suitability | Segmentation, lesion detection |

---

## Role in Second Look Pipeline

CBIS-DDSM serves as the primary source of high-quality ROI annotations in the Second Look pipeline. Its pixel-level segmentation masks allow the model to learn not just whether an image contains a suspicious finding, but precisely where within the image that finding is located. This is critical for the app's goal of highlighting regions of interest to the user rather than simply returning a binary classification. CBIS-DDSM is used specifically to guide the segmentation component of the model, complementing RSNA which provides large-scale exam-level labels and VinDR which provides bounding box annotations and BI-RADS scores. Because CBIS is derived from the original DDSM, raw DDSM cannot serve as a validation or test source while CBIS appears in training, or vice versa, due to data leakage risk.

---

## Biases

- **Abnormal-case emphasis**: CBIS-DDSM primarily contains abnormal mammograms — including benign and malignant findings — with segmentation masks. Normal screening mammograms are not meaningfully represented in this subset. This means models trained solely on CBIS-DDSM may not adequately learn the characteristics of truly normal screening exams.

- **Skewed distribution in original DDSM**: The original DDSM exhibits a heavily imbalanced distribution with 695 normal findings and 1,784 abnormal cases. This imbalance carries over when combining CBIS-DDSM with the original DDSM and requires a sub-sampling strategy to achieve a realistic class distribution. Even with sub-sampling, there are not enough normal cases to fully balance the dataset, which can bias model predictions toward abnormal findings.

---

## Limitations

- **Very limited normal cases**: The very limited representation of normal mammograms means this dataset cannot be used alone for binary cancer/no-cancer classification. Any model trained exclusively on CBIS-DDSM would need to be combined with a dataset containing normal cases, such as RSNA, before it can reliably distinguish healthy from unhealthy scans.

- **Data leakage risk**: Since CBIS-DDSM is derived from the original DDSM, the two datasets cannot both appear in training and validation/test splits simultaneously. Using both without careful separation would result in data leakage, where the model indirectly sees test data during training, artificially inflating performance metrics.

- **Small dataset size**: At 10,239 images from 2,620 patients, CBIS-DDSM is significantly smaller than RSNA (54,710 images) or VinDR (20,000 images). This limits its use as a standalone training set for deep learning models that typically require large volumes of data to generalize well.

- **Sub-sampling required**: A sub-sampling strategy is needed to achieve a realistic class distribution, and even with sub-sampling there are not enough normal cases. This makes it difficult to train models that reflect real-world screening populations where the vast majority of cases are normal.

- **Over-specification risk**: There is a risk that algorithms trained heavily on CBIS-DDSM become over-specified to this dataset distribution and exhibit degraded performance when evaluated on external screening populations with substantially different class distributions.

---

## Ethical Considerations

CBIS-DDSM contains de-identified patient mammography data collected for research purposes. As a medical imaging dataset, it must be used responsibly and solely for legitimate scientific research. The Second Look project uses this data exclusively for the purpose of developing a privacy-preserving screening tool. No attempt is made to re-identify any individual patient. All team members working with this data are expected to handle it with care and in accordance with applicable data use agreements.
