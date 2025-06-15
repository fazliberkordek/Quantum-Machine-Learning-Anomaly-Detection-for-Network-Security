# Anomaly Detection on Network Traffic with Quantum Machine Learning

## Overview
This project implements a quantum machine learning model for anomaly detection in network traffic using TensorFlow Quantum and Cirq. It demonstrates a Variational Quantum Classifier (VQC) for binary classification of network traffic as normal or anomalous.

## ⚠️ Platform Compatibility Notice
**TensorFlow Quantum is not supported natively on Apple Silicon (M1/M2/M3) Macs.**

- If you are using an Apple Silicon Mac, you will not be able to install TensorFlow Quantum directly via pip.
- The recommended way to run this project is via [Google Colab](https://colab.research.google.com/), which provides a compatible environment and free access to GPUs.

## Running on Google Colab
1. **Open Google Colab:**
   - Go to [https://colab.research.google.com/](https://colab.research.google.com/)

2. **Upload the Project Files:**
   - Click on the folder icon in the left sidebar.
   - Click the upload icon and upload `main.py` and `requirements.txt` from this repository.

3. **Install Dependencies:**
   - At the top of your Colab notebook, run the following cell to install all required packages:
     ```python
     !pip install -r requirements.txt
     ```

4. **Run the Main Script:**
   - In a new cell, run:
     ```python
     !python main.py
     ```

5. **(Optional) Edit and Experiment:**
   - You can edit `main.py` directly in Colab or upload new versions as needed.

## Project Structure
- `main.py` — Main script containing the quantum anomaly detection pipeline.
- `requirements.txt` — List of required Python packages.

## Dataset
- The script will generate a synthetic sample dataset (`CICIDS2017_sample.csv`) if not present.

## License
MIT
