# Dataset Card: VinDr-Mammo

## Overview

VinDr-Mammo is a large-scale full-field digital mammography dataset gathered in Vietnam from 5,000 mammography exams. Each patient sample contains two medial lateral oblique and two cranial caudal views, resulting in 20,000 images in total. Unlike RSNA which provides exam-level labels only, VinDr-Mammo includes detailed radiologist annotations covering BI-RADS scores, reported breast density, and bounding box locations of non-benign findings along with their category and BI-RADS assessment. The dataset used a double-read protocol where two radiologists independently reviewed each case, with disagreements resolved through arbitration. Images were originally in DICOM format and have been converted to PNG for use in this project, reducing storage requirements substantially after conversion.

---

## Dataset Details

| Field | Details |
|---|---|
| Access | [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) — requires data use agreement |
| Number of Exams | 5,000 |
| Number of Images | 20,000 (2 MLO + 2 CC views per patient) |
| Image Format | DICOM (converted to PNG, approximately 62 GB reduced from 338 GB) |
| Annotations | BI-RADS score, breast density, bounding boxes, finding category, location |
| Task Suitability | Lesion detection, localization, BI-RADS classification, breast density assessment |

---

## Role in Second Look Pipeline

VinDr-Mammo provides modern FFDM breast-level and lesion-level supervision in the Second Look pipeline. Its bounding box annotations allow the model to learn where within the image suspicious findings are located, complementing the pixel-level masks from CBIS-DDSM. Its BI-RADS scores map directly to the project's three-tier concern classification system — Low, Moderate, and Elevated — making it the most directly aligned dataset with the app's output format. It complements RSNA which provides large-scale exam-level cancer labels and CBIS-DDSM which provides high-quality ROI segmentation masks.

---

## Biases

- **Geographic bias**: VinDr-Mammo was gathered exclusively in Vietnam from a specific patient population. This may limit generalizability to patients from other countries or ethnic backgrounds where breast tissue density and cancer presentation patterns differ from those observed in the Vietnamese screening population.

- **Not biopsy confirmed**: The positive cases do not appear to be biopsy confirmed and thus may contain some false positives. This introduces uncertainty in the ground truth labels and means some cases labeled as positive findings may not represent true malignancies.

---

## Limitations

- **False positives possible**: Since positive findings are based on radiologist assessment rather than pathology confirmation through biopsy, some labeled positive cases may be incorrect. This could introduce noise into model training and affect the reliability of the learned representations.

- **Restricted access**: Access requires signing a PhysioNet data use agreement before the dataset can be used. This limits reproducibility for external researchers who have not completed this process and adds an administrative step before the data can be accessed.

- **Large storage requirement**: The original unconverted dataset is 338 GB. Even after conversion to PNG the dataset remains large at approximately 62 GB, requiring cloud-based storage and processing workflows.

- **Single-country screening population**: VinDr-Mammo was collected entirely in Vietnam under a specific screening protocol. Findings and radiologist conventions from this population may not fully generalize to screening populations in other countries or healthcare systems.

---

## Ethical Considerations

VinDr-Mammo contains de-identified patient mammography data collected for research purposes and is hosted on PhysioNet. Access requires signing a formal data use agreement which obligates all users to protect patient privacy, use the data solely for scientific research, and not attempt to re-identify any individual. The Second Look project uses this data exclusively for the purpose of developing a privacy-preserving screening tool. All team members working with this data have signed the PhysioNet data use agreement and are expected to handle it in full compliance with its terms.
