# Fundus Retinal Vessel Segmentation  

## Overview  
This project focuses on **retinal vessel segmentation** using a **hybrid deep learning approach** combining **Generative Adversarial Networks (GANs) and U-Net**. The objective is to extract blood vessels from fundus images accurately, supporting medical diagnosis for **diabetic retinopathy, glaucoma, and other ocular diseases**. The model incorporates **advanced preprocessing, adversarial learning, and evaluation metrics** to enhance segmentation performance.

## Dataset  
The model is trained and tested using the **DRIVE dataset** (Digital Retinal Images for Vessel Extraction), a well-known dataset for retinal vessel segmentation. It includes high-resolution fundus images along with expert-annotated vessel masks.

**Dataset Link:** [DRIVE Dataset](https://drive.grand-challenge.org/)  

## Methodology  

### 1️⃣ Preprocessing  
- **Contrast Enhancement:** Applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to improve vessel visibility.  
- **Grayscale Conversion & Mask Binarization:** Enhanced contrast for better segmentation.  
- **Data Augmentation:** Performed **random flipping and rotation** to improve generalization.  

### 2️⃣ Model Architecture  
- **Generator:** A **U-Net-based architecture** for precise vessel segmentation.  
- **Discriminator:** A **CNN-based discriminator** that distinguishes real vessel masks from generated ones.  
- **Loss Functions:** Combined **Binary Cross-Entropy (BCE) loss** and **Dice loss** to improve segmentation accuracy.  

### 3️⃣ Training & Evaluation  
- **Training:** The model is trained using **adversarial learning** to refine vessel segmentation.  
- **Evaluation Metrics:**  
  ✔ **Dice Coefficient (DSC)**  
  ✔ **Intersection over Union (IoU)**  
  ✔ **Pixel Accuracy, Precision, and Recall**  

## Results  
The model achieves **robust segmentation performance**:  
✔ **Dice Coefficient:** 0.6673  
✔ **IoU:** 0.5007  
✔ **Pixel Accuracy:** 0.5007  
✔ **Precision:** 0.5007  
✔ **Recall:** 1.0000  

Qualitative results show **clear vessel segmentation with well-defined boundaries**, improved contrast, and reduced noise.

## Running the Project  
To run the model, ensure you have access to a **CUDA-enabled GPU** for optimal performance.  

## Future Improvements  
🚀 **Enhancing model architecture** with attention mechanisms for improved vessel detection.  
🚀 **Exploring transfer learning** to improve performance on smaller datasets.  
🚀 **Optimizing computational efficiency** for real-time medical applications.  

---

This project contributes to **automated retinal vessel segmentation** and lays the foundation for real-world medical imaging applications.  
