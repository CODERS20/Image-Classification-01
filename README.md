# Image Classification Project

## Model Evaluation & Performance Metrics

The final ResNet-50 framework was evaluated on an independent test set containing **152 distinct images**. The evaluation metrics below were generated using the `scikit-learn` library (`metrics.classification_report`), showing an overall **Test Set Accuracy of 90.13%.**

### Classification Report Matrix

| Target Class | Precision | Recall | F1-Score | Sample Size |
| :--- | :---: | :---: | :---: | :---: |
| **Planet** | `0.9697` | `1.0000` | **`0.9846`** | 32 |
| **Elliptical Galaxy** | `0.9750` | `0.9512` | `0.9630` | 41 |
| **Star Cluster** | `0.9630` | `0.8125` | `0.8814` | 32 |
| **Nebula** | `0.8140` | `0.8974` | `0.8537` | 39 |
| **Spiral Galaxy** | `0.5556` | `0.6250` | `0.5882` | 8 |
| | | | | |
| **Overall Accuracy** | — | — | **`0.9013`** | **152** |
| **Macro Average** | `0.8554` | `0.8572` | `0.8542` | 152 |
| **Weighted Average** | `0.9080` | `0.9013` | `0.9026` | 152 |

---

## Confusion Matrix
![Confusion Matrix](final.png)

---

To keep this repository lightweight and prevent storage bloat, the full 1,508-image training batch is hosted externally at https://drive.google.com/drive/folders/1ACld4iuCli1RlKnefHQkwO6yMEXILImO?usp=sharing

A curated sub-folder (`dataset-sample/`) containing a few diagnostic preview images per class is included directly in this repository for immediate model testing and inference verification.

## Dataset Sources Used
A total of 1,508 images were used to train and test the model. The dataset is evenly split across 5 categories.
* **(https://noirlab.edu/public/images/archive/category/starclusters/page/2/?sort=-release_date)**
* **(https://www.kaggle.com/datasets/akhileshravi/nebula-images)**
* **(https://www.kaggle.com/datasets/robertmifsud/resized-reduced-gz2-images)**
* **(https://universe.roboflow.com/fashion-8zzww/planet-2vlvi/dataset/2)**
* **(https://esahubble.org/)**

---

## Model Architecture 

The core of this computer vision project uses a manually built **ResNet-50 (Residual Network)** architecture designed from scratch in PyTorch. The network is structured to prevent the vanishing gradient problem, allowing it to learn deep and complex patterns safely.

### Core Architecture Highlights

* **ResNet-50 Bottleneck Structure:** Instead of standard layers, this network uses a 3-layer bottleneck design inside each residual block. It uses a $1\times1$ convolution to shrink channels, a $3\times3$ convolution to look at features, and another $1\times1$ convolution to restore the dimensions. This makes the model incredibly deep while keeping it computationally efficient.
* **Dimensional Alignment of Tensors:** As data flows deeper into the network, the image grid size shrinks and the channel depth expands. Because you cannot add two matrices of different sizes together, a custom $1\times1$ 2D Convolutional shortcut path (`identity_downsample`) was built. This automatically resizes the shortcut path tensor so it perfectly matches the main path for matrix addition.
* **Final Output Stage:** At the very end of the network, an Adaptive Average Pooling layer flattens the features down to a 2048-dimensional vector. From there, a single linear layer is used (`nn.Linear(2048, 5)`) to map those final features directly to our 5 target classes.

---
## Setup

1. Clone this project repository and satisfy the local environment library stack requirements:
   ```bash
   pip install -r requirements.txt
