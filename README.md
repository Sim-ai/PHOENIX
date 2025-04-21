# PHOENIX: Physics-Informed Hybrid Optimization Framework for Efficient Neural Intelligence in Manufacturing
This repository provides the implementation of PHOENIX, a universal hybrid optimization framework that integrates physics-informed modeling and neural intelligence. It is designed for efficient, generalizable prediction and modeling in robotic Variable Polarity Plasma Arc (VPPA) welding applications.

- Time-ahead prediction of instability states in Variable Polarity Plasma Arc (VPPA) welding using data captured by machine vision systems (LSTM-MLP)

- Physics-constrained data modeling using quasistatic process features such as EN/EP currents and ion gas flow rates, enabling the prediction of features that would otherwise require expensive acquisition equipment (CBN-BPNN)

- Cross-condition model adaptation via incremental learning, utilizing experience replay and layer freezing strategies to automatically adjust model parameters and improve generalizability under varying operating scenarios (Incremental Learning-Based LSTM-MLP)


## 🧩 VGG16-UNet Vision Module
To obtain clearer X-ray and melt pool image boundaries, a machine vision approach was first employed to segment key regions of the images and perform binarization for downstream feature engineering. For detailed usage instructions, please refer to the GitHub repository: https://github.com/khanhha/crack_segmentation. We provide labeled X-ray and melt pool images along with corresponding JSON annotation files for the VGG16-UNet model. In actual implementation, feature extraction was performed via a locally deployed program connected to Daheng’s Mercury series industrial cameras. For detailed hardware information, please refer to the Daheng official websit https://www.daheng-imaging.com/e.

Reference: https://github.com/khanhha/crack_segmentation


## 🧠 LSTM-MLP Time-Ahead Prediction Model
To achieve time-ahead perception of welding process data, an LSTM-based architecture with a sliding window mechanism and a multilayer perceptron (MLP) is employed to perform predictive classification based on temporal information. For local evaluation, we provide offline testing code along with datasets in CSV format. To facilitate model transfer to other tasks, pretrained model parameters and  training scripts are also included. Users are encouraged to adapt the model structure to their specific applications and design their own validation strategy, such as hold-out validation or k-fold cross-validation.


## ♻️ Incremental Learning-Based LSTM-MLP for Time-Ahead Prediction
We adopt two strategies—experience replay and LSTM layer freezing—to preserve knowledge from both old and new datasets during model retraining. Specifically, data collected under new working conditions is combined with historical data in varying proportions to construct a balanced retraining dataset. This allows the model to acquire new features while maintaining previously learned capabilities. Additionally, drawing inspiration from transfer learning, selected layers of the model are frozen while others are updated, achieving a trade-off between stability and adaptability in capturing both old and new data characteristics.


## 📦 Condition-Based Neuromodulation for Physics-Informed BPNN Modeling
To address scenarios where certain input information is missing, the original model's predictive capability is preserved by leveraging condition-based constraints to infer the missing data. Compared to direct constraint methods, this approach results in improved model performance. During training, Condition-Based Neuromodulation (CBN) is applied to regulate the model’s output, enabling the effective incorporation of condition information into the learning process through dynamic modulation.

## Dependencies
This project requires the following Python packages:

Package | Version
torch | 1.8.0
pandas | 2.0.3
numpy | 1.24.4
scikit-learn | 1.3.2
matplotlib | 3.2.2
seaborn | 0.13.2



📌 Citation
If you find this work helpful, please cite our project (DOI to be added after publication).
