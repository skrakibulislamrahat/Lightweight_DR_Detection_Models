🧠 Evaluation of Lightweight AI Models for Early Detection of Diabetic Retinopathy
This repository contains code, results, and documentation for our study titled:
“Evaluation of Lightweight AI Models for Early Detection of Diabetic Retinopathy in Mobile-Friendly Clinical Settings.”

📄 The study benchmarks three lightweight convolutional neural networks—EfficientNetB0, MobileNetV2_100, and SqueezeNet1_0—on the APTOS 2019 Blindness Detection dataset, focusing on binary classification of referable vs. non-referable diabetic retinopathy (DR) cases.

🔍 Research Highlights
🧪 Dataset: APTOS 2019

📊 Models: EfficientNetB0, MobileNetV2_100, SqueezeNet1_0

🧠 Task: Binary DR classification (0–1 → Non-referable, 2–4 → Referable)

📈 Top Accuracy: EfficientNetB0 at 93.52% (F1: 0.9221, AUC: 0.9331)

⚡ Deployability Focus: Emphasis on real-time mobile/low-resource use

✅ Reproducibility: Results are consistent, reproducible, and documented

🗂 Repository Structure
graphql
Copy
Edit
Lightweight_DR_Detection_Models/
├── models/             # CNN model definitions (EfficientNetB0, MobileNetV2, etc.)
├── notebooks/          # Training, evaluation, and visualization notebooks
│   ├── training.ipynb
│   ├── evaluation.ipynb
│   └── metrics_visualization.ipynb
├── results/            # ROC curves, confusion matrices, bar charts
├── utils/              # Helper functions for preprocessing and metrics
├── requirements.txt    # Python package dependencies
└── README.md           # You're here!
🛠️ How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/skrakibulislamrahat/Lightweight_DR_Detection_Models.git
cd Lightweight_DR_Detection_Models
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset:

From: APTOS 2019 on Kaggle

Place inside: data/ directory (not included due to license)

Run training and evaluation:

Use notebooks/training.ipynb for model training

Use notebooks/evaluation.ipynb for metrics and plots

📊 Results Summary
Model	Accuracy	F1-Score	AUC	Inference Time
EfficientNetB0	93.52%	0.9221	0.9331	35 ms
MobileNetV2_100	93.21%	0.9153	0.9285	28 ms
SqueezeNet1_0	92.64%	0.9029	0.9220	19 ms

📌 Inference time was measured using Google Colab Pro+ with NVIDIA Tesla T4 GPUs.

📁 Citation
If you use this code or cite the findings, please reference the paper:

SK Rakib Ul Islam Rahat, et al.
"Evaluation of Lightweight AI Models for Early Detection of Diabetic Retinopathy in Mobile-Friendly Clinical Settings"
(Submitted to Computer Science and Information Systems – ComSIS, 2025)

🤝 Contact
For questions, please contact:
📧 skrakibulislamrahat@gmail.com
