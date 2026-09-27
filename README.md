
This project is a Self-Supervised Learning (SSL) computer vision framework built in PyTorch.

What is Self-Supervised Learning?
Self-supervised learning is a type of machine learning where a model learns powerful visual features from unlabelled images (without needing humans to manually tag or label thousands of pictures). It does this by creating clever "puzzles" or tasks for the network to solve on its own.

What this specific project does:
Pre-trains Visual Backbones: It takes a neural network architecture and trains it on raw images by combining two powerful self-supervised strategies:

Clustering-based objectives (similar to SwAV): Grouping similar image features together without labels.

Spatial transformation tasks (like RotNet): Forcing the network to predict things like the rotation angle of an image, which teaches it to understand shapes, structures, and spatial orientation.

Evaluates Representations: Once the model finishes pre-training, it includes evaluation pipelines (like linear evaluation) to test how well the learned features can be used for downstream tasks (like image classification) by freezing the main network and training a simple classifier on top.

In short, it's a complete pipeline to train robust computer vision models from scratch using unlabelled data and test how smart those models have become.

# Self-Supervised Visual Representation Learning Framework

A PyTorch-based implementation for training and evaluating self-supervised visual representation models using combination techniques (such as clustering-based objectives integrated with spatial transformation tasks).

---

## 📂 Project Structure

```text
├── scripts/           # Execution scripts for training and downstream evaluation
├── src/               # Core codebase (models, losses, and custom data augmentations)
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
⚙️ Requirements & Installation
This project requires Python and PyTorch. Follow the steps below to set up your environment:

1. Clone the Repository
Bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create a Virtual Environment
Bash
conda create -n ssl-env python=3.8.8
conda activate ssl-env
3. Install Dependencies
Install the required packages matching your local CUDA configuration:

Bash
pip install -r requirements.txt
(Note: Ensure you install the appropriate PyTorch version corresponding to your GPU hardware/CUDA version.)

🏃‍♂️ Usage & Workflow
All execution pipelines are managed via shell scripts located in the scripts/ directory.

Step 1: Model Pre-training
To run pre-training configurations (including baseline setups and auxiliary task combinations like SwAV + RotNet):

Bash
cd scripts/
bash train_swav_rotnet.sh
Step 2: Linear Evaluation
To evaluate the quality of the learned representation using a frozen backbone:

Bash
bash linear_eval.sh
📄 License
Distributed under the MIT License. See LICENSE for more information. Portions of this codebase adapt components from open-source vision repositories.


