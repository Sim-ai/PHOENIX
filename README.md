# PHOENIX: Physics-Informed Hybrid Optimization Framework for Efficient Neural Intelligence in Manufacturing
This repository provides the implementation of PHOENIX, a universal hybrid optimization framework that integrates physics-informed modeling and neural intelligence. It is designed for efficient, generalizable prediction and modeling in robotic Variable Polarity Plasma Arc (VPPA) welding applications.

- Time-ahead prediction of instability states in Variable Polarity Plasma Arc (VPPA) welding using data captured by machine vision systems

- Physics-constrained data modeling using quasistatic process features such as EN/EP currents and ion gas flow rates, enabling the prediction of features that would otherwise require expensive acquisition equipment

- Cross-condition model adaptation via incremental learning, utilizing experience replay and layer freezing strategies to automatically adjust model parameters and improve generalizability under varying operating scenarios



## 🧩 VGG16-UNet Vision Module
To obtain clearer X-ray and melt pool image boundaries, a machine vision approach was first employed to segment key regions of the images and perform binarization for downstream feature engineering. For detailed usage instructions, please refer to the GitHub repository: https://github.com/khanhha/crack_segmentation. We provide labeled X-ray and melt pool images along with corresponding JSON annotation files for the VGG16-UNet model. In actual implementation, feature extraction was performed via a locally deployed program connected to Daheng’s Mercury series industrial cameras. For detailed hardware information, please refer to the Daheng official websit https://www.daheng-imaging.com/e.


## 🧠 LSTM-MLP Time-Ahead Prediction Model
To achieve time-ahead perception of welding process data, an LSTM-based architecture with a sliding window mechanism and a multilayer perceptron (MLP) is employed to perform predictive classification based on temporal information. For local evaluation, we provide offline testing code along with datasets in CSV format. To facilitate model transfer to other tasks, pretrained model parameters and  training scripts are also included. Users are encouraged to adapt the model structure to their specific applications and design their own validation strategy, such as hold-out validation or k-fold cross-validation.

## ♻️ Incremental Learning-Based LSTM-MLP for Time-Ahead Prediction













🧠 Physics-Infused Modeling via Condition-Weighted BPNN
Integrates quasistatic physical parameters (e.g., EN/EP current and ion gas flow) as conditioning inputs to modulate network outputs. This strategy replaces traditional Conditional Batch Normalization (CBN) with Condition-Based Neuromodulation, embedding physics-derived constraints directly into the network training process.

♻️ Incremental Learning for Cross-Scenario Adaptability
Supports automatic parameter tuning using experience replay and layer freezing strategies. Enhances the model's robustness and generalization under changing operating conditions and new data streams.

📦 Module Descriptions
1. Visual Segmentation for Melt Pool Feature Extraction
Model: VGG16-UNet

Function: Performs image segmentation on melt pool and X-ray images for key region extraction and binary mask generation.

Resources:

Sample dataset (X-ray and melt pool images)

JSON-format masks

Pretrained weights

Hardware: Supports integration with DaHeng Mercury series cameras (demo program provided).

Reference: github.com/xxx/vgg16-unet-xray

2. LSTM-MLP for Time-Ahead Prediction
Architecture: LSTM with sliding window → MLP

Task: Sequence-based classification of future melt pool stability.

Resources:

Offline testing code (CSV input)

Pretrained models

Note: Model structure is modular and transferable to other datasets/tasks.

3. Incremental Learning with Experience Replay and Layer Freezing
Strategies:

Mixes new and historical data in user-defined proportions.

Freezes bottom layers of LSTM to retain prior knowledge.

Result: Balances retention of prior knowledge with adaptation to new data.

4. Condition-Constrained BPNN for Physics-Infused Data Modeling
Goal: Predict high-cost physical features using only low-cost sensor data, guided by physical welding conditions.

Method: Conditioning inputs (e.g., EN current, EP current, gas flow) are used to modulate network output:

𝑤
𝑒
𝑖
𝑔
ℎ
𝑡
𝑠
=
𝑆
𝑖
𝑔
𝑚
𝑜
𝑖
𝑑
(
𝑊
𝑐
𝑜
𝑛
𝑑
×
[
𝐼
𝐸
𝑁
,
𝐼
𝐸
𝑃
,
𝑄
]
+
𝑏
𝑐
𝑜
𝑛
𝑑
)
𝑜
𝑢
𝑡
𝑝
𝑢
𝑡
=
𝐵
𝑃
𝑁
𝑁
(
𝑥
)
×
(
1
+
𝑤
𝑒
𝑖
𝑔
ℎ
𝑡
𝑠
)
weights=Sigmoid(W 
c
​
 ond×[I 
E
​
 N,I 
E
​
 P,Q]+b 
c
​
 ond)output=BPNN(x)×(1+weights)
Advantage: Introduces physical constraints during training without requiring expensive data at inference time.

📁 Data & Access
📸 X-ray and Melt Pool Imaging Dataset
Real-time tungsten tracer-enhanced X-ray imaging of melt pool flow fields.
🔗 [Available at: xxx]

📂 Code-Compatible Dataset
The dataset used by all models is included with this repository.
🔗 [Available at: xxx]

🧪 Experimental Setup Overview
Material: 5052 Aluminum Alloy (100 × 20 × 5 mm)

X-ray Source: 230 kV, 1.5 mA

High-Speed Camera: 1000 FPS, 800 × 600 px, FOV: 22 × 20 mm

Tracer: Tungsten particles (0.03 mm diameter)

📌 Citation
If you find this work helpful, please cite our project (DOI to be added after publication).
