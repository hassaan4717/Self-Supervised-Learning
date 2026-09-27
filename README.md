 [[Poster](https://sslneurips21.github.io/files/Poster/Paper_id_25.pdf)][[Paper](https://sslneurips21.github.io/files/CameraReady/SSLW_upload.pdf)].
Markdown
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


