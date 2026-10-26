# Dataset Card: RSNA Screening Mammography

## Overview

The RSNA Screening Mammography Breast Cancer Detection dataset contains full-field digital mammography images from a large-scale screening population covering nearly 12,000 patients and over 54,000 individual mammogram screens. The dataset provides exam-level cancer labels rather than lesion-level annotations, making it particularly well-suited for large-scale classification and cancer risk learning tasks. Images were originally in DICOM format and have been converted to PNG for use in this project, resulting in a dataset of approximately 144 GB. Its scale makes it the primary dataset for large-scale screening classification within the Second Look pipeline.

---

## Dataset Details

| Field | Details |
|---|---|
| Access | [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) / [AWS Open Data](https://registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection/) |
| Number of Patients | 11,914 |
| Number of Images | 54,710 |
| Image Format | DICOM (converted to PNG, 144 GB) |
| Annotations | Exam-level cancer label |
| Task Suitability | Exam-level classification, cancer risk learning |

---

## Positive Case Breakdown

| Category | Count |
|---|---|
| Right breast positive | 570 |
| Left breast positive | 588 |
| Positive in both breasts | 2 |
| Total positive rate | 2.1% |

---

## Role in Second Look Pipeline

RSNA is the primary dataset for large-scale exam-level cancer and risk learning in the Second Look pipeline. Its scale makes it particularly useful for training deep learning models on large screening populations. Because it reflects a real screening population where the vast majority of cases are normal, it provides the negative examples that CBIS-DDSM lacks. It complements CBIS-DDSM which provides high-quality ROI segmentation masks, and VinDR which provides modern FFDM lesion-level supervision and BI-RADS scores.

---

## Biases

- **Extreme class imbalance**: Only 2.1% of cases are positive, meaning the dataset is heavily skewed toward normal findings. This reflects the realistic distribution of a screening population but creates significant class imbalance challenges during model training.

- **Screening population bias**: The dataset reflects a screening population rather than a diagnostic population. This means it skews heavily toward normal findings, which is realistic for deployment but requires careful handling during training to ensure the model does not underperform on the rare positive cases that matter most clinically.

---

## Limitations

- **No lesion-level annotations**: Unlike CBIS-DDSM or VinDR, RSNA does not provide bounding boxes or segmentation masks. It cannot be used alone for lesion localization or segmentation training. The model can learn whether cancer is present at the exam level but cannot learn where within the image to look.

- **Large storage requirement**: At 144 GB after PNG conversion, the dataset often requires cloud-based storage and processing workflows due to its size.

- **Exam-level labels only**: Labels are provided at the exam level, not the lesion level. This limits fine-grained analysis and means RSNA must be combined with datasets like CBIS-DDSM or VinDR for any localization or segmentation task.

- **Very low positive rate**: At only 2.1% positive cases, standard training without rebalancing will produce a model biased toward predicting negative, potentially missing the rare positive cases that are most clinically significant.

---

## Ethical Considerations

RSNA contains de-identified patient mammography data collected for research purposes through the Radiological Society of North America. As a medical imaging dataset, it must be used responsibly and solely for legitimate scientific research. The Second Look project uses this data exclusively for the purpose of developing a privacy-preserving screening tool. No attempt is made to re-identify any individual patient. All team members working with this data are expected to handle it with care and in accordance with applicable data use agreements.
