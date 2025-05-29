# Quantum Anomaly Detection using Qiskit and CICIDS2017 Dataset

This project applies **Quantum Machine Learning** to detect anomalies in network traffic using the **CICIDS2017 dataset**. It leverages **Qiskit Machine Learning** tools, and is compatible with the latest versions of `qiskit` and `qiskit-machine-learning`.

## 🧠 What It Does

The project builds a **Variational Quantum Classifier (VQC)** that:

* Takes network traffic features as input,
* Encodes them into quantum states,
* Uses a parameterized quantum circuit (ansatz) to learn patterns,
* Classifies network flows as **normal (BENIGN)** or **anomalous (attack)**.

## 📁 Files

* `helper_notebook.ipynb`: Main training and visualization notebook
* `CICIDS2017_sample.csv`: Sample preprocessed version of the dataset
* `README.md`: Project overview and instructions

## 🛠️ Dependencies

Install the required packages using pip:

```bash
pip install qiskit qiskit-machine-learning pandas scikit-learn matplotlib seaborn
```

## 🚀 How to Run

1. Place `CICIDS2017_sample.csv` in the root directory.
2. Open the notebook `helper_notebook.ipynb`.
3. Run cells in order:

   * Data loading and preprocessing
   * Quantum model construction (VQC)
   * Training and evaluation
   * PCA visualization of predictions

## 📊 Dataset Features Used

The notebook uses the following features from the CICIDS2017 dataset:

* `Flow Duration`
* `Total Fwd Packets`
* `Total Backward Packets`
* `Flow Bytes/s`
* `Flow Packets/s`
* `Fwd Packet Length Mean`

Label is mapped to binary:

* `BENIGN` → 0
* All other types → 1 (anomaly)

## 🧪 Results

* Outputs training and testing accuracy
* Displays PCA visualization comparing predicted vs true labels

## 📈 Future Enhancements

* Use a larger subset or full CICIDS2017
* Add confusion matrix and precision/recall metrics
* Run on IBM Quantum real devices
* Explore unsupervised anomaly detection (e.g., quantum kernel methods)

---

Created with ❤️ using Qiskit by \[Fazli Berk Ordek].
