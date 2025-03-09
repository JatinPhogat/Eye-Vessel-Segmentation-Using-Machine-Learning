# 🏥 Fundus Retinal Vessel Segmentation using GANs

## 📌 Overview  
This project implements **retinal vessel segmentation** using a **hybrid deep learning model** combining **Generative Adversarial Networks (GANs) and U-Net**. The goal is to accurately extract blood vessels from fundus images, aiding in medical diagnoses for **diabetic retinopathy, glaucoma, and other ocular diseases**. The approach integrates **advanced preprocessing, adversarial learning, and evaluation metrics** to enhance segmentation performance.

## 📂 Dataset  
The model is trained and tested using the **DRIVE dataset** (Digital Retinal Images for Vessel Extraction), a widely used dataset for retinal vessel segmentation. It includes high-resolution fundus images along with expert-annotated vessel masks.

📌 **Dataset Link:** [DRIVE Dataset](https://drive.grand-challenge.org/)  

## 🛠️ Methodology  
### 1️⃣ Preprocessing  
 **Contrast Enhancement:** Applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to improve vessel visibility.  
 **Grayscale Conversion & Mask Binarization:** Enhanced contrast for better segmentation.  
 **Data Augmentation:** Performed **random flipping and rotation** to improve generalization.  

### 2️⃣ Model Architecture  
 **Generator:** A **U-Net-based architecture** for precise vessel segmentation.  
 **Discriminator:** A **CNN-based discriminator** that distinguishes real vessel masks from generated ones.  
 **Loss Functions:** Combined **Binary Cross-Entropy (BCE) loss** and **Dice loss** to improve segmentation accuracy.  

### 3️⃣ Training & Evaluation  
✔ **Training:** The model is trained using **adversarial learning** to refine vessel segmentation.  
✔ **Evaluation Metrics:**  
   -  **Dice Coefficient (DSC)**  
   -  **Intersection over Union (IoU)**  
   -  **Pixel Accuracy, Precision, and Recall**  

## 📊 Results  
The model achieves **robust segmentation performance**:  
 **Dice Coefficient:** 0.6673  
 **IoU:** 0.5007  

### Steps to Run:  
1️⃣ Clone the repository:
```bash
git clone https://github.com/JatinPhogat/Eye-Vessel-Segmentation-Using-Machine-Learning/
```

2️⃣ Run on jupyter notebook:
```bash
jupyter notebook
```
Open and execute `fundus_segmentation.ipynb`.

---
This project contributes to **automated retinal vessel segmentation** and lays the foundation for real-world medical imaging applications. 🎯
