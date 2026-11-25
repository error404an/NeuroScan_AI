NEUROSCAN AI - MRI MODEL MODULE
================================

SAFETY
------
Educational/research prototype only. It is not a medical device and must not
diagnose, rule out, or guide treatment for any person. A dataset score is not
evidence of clinical accuracy.

DATASET
-------
Suggested dataset: BDNeuro-MRI, Mendeley Data, Version 7 (CC BY 4.0):
https://data.mendeley.com/datasets/zwr4ntf94j/7

Read the dataset page, license, provenance and limitations before downloading.
Cite the dataset authors in your project report.

Expected folder layout:
NeuroScan_AI/
  dataset/
    Brain_Tumor_MRI_Dataset_Final/
      train/
        glioma/
        meningioma/
        pituitary/
        no_tumor/
      val/
        glioma/
        meningioma/
        pituitary/
        no_tumor/
      test/
        glioma/
        meningioma/
        pituitary/
        no_tumor/

STEP 1 - INSTALL PACKAGES
-------------------------
Activate your existing virtual environment:
  .venv\\Scripts\\activate

Then run:
  python -m pip install --upgrade pip
  pip install tensorflow scikit-learn matplotlib

STEP 2 - EXTRACT DATASET
------------------------
Create dataset/ in the project root. Extract the downloaded archive so the
train, val and test folders are directly inside:
dataset/Brain_Tumor_MRI_Dataset_Final/

Do not mix train, val and test. Keep the four class folders as named above.

STEP 3 - TRAIN
--------------
From the project root:
  python scripts/train_model.py

The model and class order will be saved in models/:
  brain_mri_model.keras
  class_names.json

Training can take time, especially on CPU. Keep the terminal open.

STEP 4 - EVALUATE
-----------------
After training:
  python scripts/evaluate_model.py

Review per-class precision, recall, F1 and the confusion matrix. Do not tune
repeatedly against the held-out test set.

NOTES
-----
The code uses TensorFlow's image_dataset_from_directory folder loader:
https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

This is a 2D image-slice classifier, not a full 3D DICOM/NIfTI MRI reader.
Dataset duplication, patient overlap, scanner differences and label quality
can distort results. Do not upload identifiable patient data.
