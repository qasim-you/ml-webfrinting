# Fingerprinting Encrypted DNS: Exploiting Metadata Leakage in DNS over QUIC

This repository contains the code, feature extraction framework, and machine learning models for the research paper **"Fingerprinting Encrypted DNS: Exploiting Metadata Leakage in DNS over QUIC"** by Marta Moure-Garrido, Celeste Campo, and Carlos Garcia-Rubio from the University Carlos III of Madrid. 

## 📖 Overview

The growing adoption of HTTP/3 and the Quick UDP Internet Connections (QUIC) protocol has enabled DNS over QUIC (DoQ), which offers major improvements in performance, latency, and encryption by leveraging TLS 1.3 and zero round-trip time (0-RTT) connection establishment. However, this project demonstrates that DoQ remains highly vulnerable to Website Fingerprinting (WF) attacks. Even though packet payloads and domain names are fully encrypted, observable traffic patterns—such as packet sizes, inter-arrival times, and stream behaviors—leak metadata that adversaries can exploit to infer a user's web activity. 

## 🗄️ Dataset

This project utilizes a real-world DoQ+QUIC web traffic dataset, originally captured by Csikor et al. using the CloudLab testbed across diverse geographic locations. 
*   **Target Websites:** The top 100 websites ordered by the Tranco rank.
*   **Data Structure:** The dataset includes packet-level traces representing individual sessions, capturing raw header metadata, packet size, inter-arrival time, and packet direction.

## ⚙️ Methodology & Pipeline

Our comprehensive evaluation framework is adapted specifically for encrypted DNS traffic:

1.  **Flow Identification:** Packets are grouped into bidirectional flows based on a unique combination of source and destination IP addresses and ports. Cumulative relative times are computed to reconstruct the temporal evolution of each trace.
2.  **Feature Extraction:** We extract a robust set of statistical features per flow, including packet counts, protocol statistics, byte counts, packet length statistics, and temporal features. 
3.  **Feature Relevance Analysis:** Using Mutual Information (MI) scores, we identify that volume and size-related features (like `received_bytes` and `recv_length_mean`) are the most discriminative.
4.  **Machine Learning Classification:** The dataset is split using an 80/20 stratified partition. We train and evaluate several supervised models, including Random Forest, XGBoost, and Extra Trees.
5.  **Explainable AI (XAI):** We use SHapley Additive exPlanations (SHAP) to interpret model predictions, confirming that features related to packet size and flow volume have the strongest positive influence on correct domain classification.
6.  **Dimensionality Reduction:** Principal Component Analysis (PCA) is applied to reduce the feature space to 9 components (retaining over 90% of the variance), allowing us to explore the trade-off between computational efficiency and classification accuracy.

## 🚀 Key Results

*   **Classification Accuracy:** XGBoost achieved the highest accuracy of 79.3% on the full 100-website dataset, while a Random Forest model achieved an average accuracy of 95% when restricted to the top-10 most frequently visited websites.
*   **Metadata Leakage:** Feature importance analysis revealed that `sent_length_mean` and `sent_bytes` are highly influential for distinguishing DoQ traffic, underscoring the characteristic size patterns of the protocol.
*   **Performance Trade-offs:** PCA successfully reduced training times for tree-based models, highlighting a viable pathway for reducing energy consumption in resource-constrained settings, albeit with a moderate reduction in accuracy.

## 🔮 Future Directions

Based on the feature-level insights uncovered by our Explainable AI models, future research utilizing this repository can focus on:
*   **"Smart" Traffic Shaping:** Developing targeted obfuscation algorithms that selectively pad or delay only the most revealing features (like `sent_length_mean` and `received_bytes`) to balance privacy with network performance.
*   **Open-World Environments:** Expanding the models to massive, open-world settings to test the resilience of fingerprinting defenses when distinguishing monitored sites from an infinite sea of unmonitored ones.
*   **Real-Time Defenses:** Leveraging our lightweight PCA-reduced models to build dynamic defense systems that monitor outgoing DoQ traffic and inject countermeasures only when a vulnerability threshold is met.

## 📝 Citation

If you use this code or dataset in your research, please cite the original paper:

> Moure-Garrido, M., Campo, C., & Garcia-Rubio, C. (n.d.). Fingerprinting Encrypted DNS: Exploiting Metadata Leakage in DNS over QUIC. Department of Telematic Engineering, University Carlos III of Madrid.
