# Quantum Anomaly Detection on CICIDS2017 with Qiskit (2024+)
# STEP 1: Install Dependencies (run in notebook shell)
# !pip install qiskit qiskit-machine-learning scikit-learn pandas matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss

algorithm_globals.random_seed = 42

# Load data
raw_data = pd.read_csv("CICIDS2017_sample.csv")
selected_features = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Mean"
]
data = raw_data[selected_features + ["Label"]].replace([np.inf, -np.inf], np.nan).dropna()
X = data[selected_features].values
y = data["Label"].apply(lambda x: 0 if x == "BENIGN" else 1).values

scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=0.8, random_state=42, stratify=y
)

num_features = X_train.shape[1]
feature_map = ZZFeatureMap(num_features, reps=1, entanglement="full")
ansatz = RealAmplitudes(num_features, reps=2, entanglement="full")
quantum_circuit = feature_map.compose(ansatz)

sampler = Sampler()
qnn = SamplerQNN(
    circuit=quantum_circuit,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    sampler=sampler,
    sparse=False,
)

classifier = NeuralNetworkClassifier(
    neural_network=qnn,
    loss=CrossEntropyLoss(),
    optimizer="cobyla",
    initial_point=np.random.rand(ansatz.num_parameters),
    one_hot=False,
)

classifier.fit(X_train, y_train)
train_acc = classifier.score(X_train, y_train)
test_acc = classifier.score(X_test, y_test)
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

predictions = classifier.predict(X_test)
print("Sample Predictions:", predictions[:10])

pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_test)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=predictions, palette='Set1', style=y_test, s=100)
plt.title("PCA Projection of Quantum Classifier Predictions")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Prediction / True Label", loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()