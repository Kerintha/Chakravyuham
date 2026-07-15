# Chakravyuham
### Behavior-Driven Edge AI Intrusion Detection for In-Vehicle Networks

> 🚧 **Work in Progress**
>
> Chakravyuham is an ongoing research and engineering project focused on building a behavior-driven Intrusion Detection System (IDS) for automotive CAN networks. This repository contains the current prototype and will continue to evolve with additional models, deployment features, and research improvements. (You can check out our planned future roadmap below, as well as in depth in our technical report in the docs/ folder.)

Developed for **Tata Technologies InnoVent '27** under the theme **AI at the Edge for Automotive Cybersecurity**.

---

## Overview

Modern vehicles contain dozens of Electronic Control Units (ECUs) that communicate over the Controller Area Network (CAN) bus. Since CAN provides neither authentication nor encryption, any compromised node can inject malicious messages that appear legitimate to other ECUs. Hackers take advantage of this vulnerability to perform malicious attacks on vehicles.

Chakravyuham investigates how machine learning can improve automotive intrusion detection beyond traditional rule-based systems while remaining practical for real-world deployment. Instead of learning dataset-specific shortcuts, the project focuses on behavioral learning, rigorous evaluation, and deployment-oriented system design.

---

## Features

- Behavioral intrusion detection for in-vehicle CAN networks
- Detection of Normal, DoS, Fuzzy, and ECU Impersonation attacks
- Leakage-aware temporal evaluation pipeline
- Behavioral feature engineering
- Comparative benchmarking across multiple ML architectures
- Deployment-oriented design for embedded automotive systems

---

## Model Architectures

The current prototype evaluates multiple approaches including:

- XGBoost
- LightGBM
- Random Forest
- Graph Convolutional Networks (GCNs)

Rather than committing to a single model, we tested across different architectures to evaluate detection capability, latency, explainability, and suitability for edge deployment. 

---

## Tech Stack

**Languages & Frameworks**
- Python
- FastAPI
- React
- TypeScript
- Node.js

**Machine Learning**
- PyTorch
- PyTorch Geometric
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

**Data Processing**
- Pandas
- NumPy

**Datasets**
- HCRL Car-Hacking Dataset
- OTIDS Dataset

---

## Technical Report

A detailed explanation of our project's motivation, methodology, benchmarking results, engineering decisions, and deployment roadmap is available in:

**docs/Chakravyuham_Technical_Report.pdf**

---

## Future Work

Planned improvements include:

- Replay attack detection (sequence-based modeling)
- Synthetic data augmentation
- Vehicle Signal Specification (VSS) integration
- OEM-specific calibration
- Hybrid Rule + ML detection
- Edge deployment on embedded automotive hardware
- VSOC-assisted model updates

---

## Current Status

This repository represents the current research prototype.

Ongoing work includes:

- Improving cross-vehicle generalization
- Expanding attack coverage
- Optimizing embedded inference
- Refining the deployment pipeline
- Improving documentation and code organization

---

## Team

Developed by:

- Abhinav Mucharla
- Pranav Raj
- Addanki Srinidhi
- Tejasri Meedimale
- Thota Bindu
- Darshitha

---

## License

This repository is intended for educational and research purposes.
