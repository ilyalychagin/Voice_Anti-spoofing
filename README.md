# Voice Anti-spoofing with LightCNN

This project implements and trains a Countermeasure (CM) system to detect spoofing attacks (replays, synthetic speech, etc.) on the Logical Access (LA) partition of the ASVSpoof 2019 Dataset. The core architecture is a LightCNN (LCNN), implemented from scratch in PyTorch following the specifications from papers: 

- **[STC Antispoofing Systems for the ASVspoof2019 Challenge](https://arxiv.org/abs/1904.05576)** by Speech Technology Center - used as the foundation for the LightCNN architecture implementation
- **[A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection](https://arxiv.org/abs/2103.11326)** - used for the training recipe and data preparation scheme

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and training purposes.

### Installing

A step-by-step series of examples to get a development environment running.

1.  **Create and activate a Conda environment:**
    ```bash
    conda create -n voice-anti-spoofing python=3.8
    conda activate voice-anti-spoofing
    ```

2.  **Clone this repository and navigate into the project directory:**
    ```bash
    git clone https://github.com/ilyalychagin/Voice_Anti-spoofing.git
    cd Voice_Anti-spoofing
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    - Download the `LA` partition from the official [ASVSpoof 2019 dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset).
    - Place the downloaded data in the root of the project directory.

5.  **Start the training process:**
    ```bash
    python3 train.py
    ```
    The script will begin training the LightCNN model on the ASVSpoof 2019 LA training set.

### Results

    * **EER:** 7% (Equal error rate)

## Acknowledgments

- Acknowledgement to the [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) for providing a structured project foundation.