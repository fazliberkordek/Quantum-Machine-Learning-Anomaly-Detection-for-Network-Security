import cirq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Tuple
import tensorflow as tf
import tensorflow_quantum as tfq

class QuantumAnomalyDetector:
    """
    Quantum Machine Learning model for network anomaly detection using Cirq.
    Implements a Variational Quantum Classifier (VQC) for binary classification.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_qubits)
        
    def create_data_encoding_circuit(self, data_point: np.ndarray) -> cirq.Circuit:
        """
        Create quantum circuit to encode classical data into quantum states.
        Uses angle encoding where data features are encoded as rotation angles.
        """
        circuit = cirq.Circuit()
        
        # Normalize data to [0, π] range for rotation angles
        normalized_data = np.arctan(data_point) + np.pi/2
        
        # Encode each feature as a rotation around Y-axis
        for i, angle in enumerate(normalized_data[:self.n_qubits]):
            circuit.append(cirq.ry(angle)(self.qubits[i]))
            
        return circuit
    
    def create_variational_circuit(self) -> Tuple[cirq.Circuit, List[cirq.Symbol]]:
        """
        Create parameterized quantum circuit (ansatz) for learning.
        Uses alternating layers of single-qubit rotations and entangling gates.
        """
        circuit = cirq.Circuit()
        symbols = []
        
        for layer in range(self.n_layers):
            # Single-qubit rotations with trainable parameters
            for i, qubit in enumerate(self.qubits):
                # Create unique symbols for each parameter
                theta_symbol = cirq.Symbol(f'theta_{layer}_{i}')
                phi_symbol = cirq.Symbol(f'phi_{layer}_{i}')
                symbols.extend([theta_symbol, phi_symbol])
                
                # Apply parameterized rotations
                circuit.append(cirq.ry(theta_symbol)(qubit))
                circuit.append(cirq.rz(phi_symbol)(qubit))
            
            # Entangling layer (except for last layer)
            if layer < self.n_layers - 1:
                for i in range(len(self.qubits) - 1):
                    circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
                # Add circular entanglement
                if len(self.qubits) > 2:
                    circuit.append(cirq.CNOT(self.qubits[-1], self.qubits[0]))
        
        return circuit, symbols
    
    def create_measurement_circuit(self) -> cirq.Circuit:
        """Create measurement circuit for readout."""
        circuit = cirq.Circuit()
        # Measure the first qubit for binary classification
        circuit.append(cirq.measure(self.qubits[0], key='result'))
        return circuit
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data with scaling and dimensionality reduction."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            X_reduced = self.pca.fit_transform(X_scaled)
        else:
            X_scaled = self.scaler.transform(X)
            X_reduced = self.pca.transform(X_scaled)
        
        return X_reduced, y
    
    def build_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build and compile the quantum neural network model."""
        # Create the variational circuit
        var_circuit, symbols = self.create_variational_circuit()
        
        # Create input placeholders for data encoding
        input_symbols = [cirq.Symbol(f'x_{i}') for i in range(self.n_qubits)]
        
        # Create data encoding circuit with symbolic parameters
        encoding_circuit = cirq.Circuit()
        for i, symbol in enumerate(input_symbols):
            encoding_circuit.append(cirq.ry(symbol)(self.qubits[i]))
        
        # Combine encoding and variational circuits
        full_circuit = encoding_circuit + var_circuit
        
        # Create readout operator (Pauli-Z measurement on first qubit)
        readout_op = cirq.Z(self.qubits[0])
        
        # Build TensorFlow Quantum model
        input_layer = tf.keras.layers.Input(shape=(self.n_qubits,), dtype=tf.float32)
        
        # Convert classical data to quantum data
        encoding_layer = tfq.layers.AddCircuit()(input_layer, append=encoding_circuit)
        
        # Apply variational quantum layer
        quantum_layer = tfq.layers.PQC(
            var_circuit,
            readout_op,
            repetitions=1000
        )(encoding_layer)
        
        # Add classical post-processing layer
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(quantum_layer)
        
        # Create and compile model
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> dict:
        """Train the quantum model."""
        # Preprocess training data
        X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train, fit=True)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self.preprocess_data(X_val, y_val, fit=False)
            validation_data = (X_val_processed, y_val_processed)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train_processed, y_train_processed)
        
        # Train the model
        history = self.model.fit(
            X_train_processed, y_train_processed,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        X_processed, _ = self.preprocess_data(X, fit=False)
        predictions = self.model.predict(X_processed)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_processed, _ = self.preprocess_data(X, fit=False)
        return self.model.predict(X_processed).flatten()


def load_and_prepare_data(file_path: str = 'CICIDS2017_sample.csv') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare the CICIDS2017 dataset for quantum processing.
    """
    # Create sample data if file doesn't exist
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Creating sample network traffic data...")
        # Generate synthetic network traffic data
        np.random.seed(42)
        n_samples = 1000
        
        # Normal traffic (70% of data)
        n_normal = int(0.7 * n_samples)
        normal_data = {
            'Flow Duration': np.random.exponential(1000, n_normal),
            'Total Fwd Packets': np.random.poisson(10, n_normal),
            'Total Backward Packets': np.random.poisson(8, n_normal),
            'Flow Bytes/s': np.random.exponential(5000, n_normal),
            'Flow Packets/s': np.random.exponential(20, n_normal),
            'Fwd Packet Length Mean': np.random.normal(500, 100, n_normal),
            'Label': ['BENIGN'] * n_normal
        }
        
        # Anomalous traffic (30% of data)
        n_anomaly = n_samples - n_normal
        anomaly_data = {
            'Flow Duration': np.random.exponential(5000, n_anomaly),  # Longer duration
            'Total Fwd Packets': np.random.poisson(50, n_anomaly),   # More packets
            'Total Backward Packets': np.random.poisson(2, n_anomaly), # Fewer responses
            'Flow Bytes/s': np.random.exponential(50000, n_anomaly),  # Higher bandwidth
            'Flow Packets/s': np.random.exponential(100, n_anomaly),  # Higher packet rate
            'Fwd Packet Length Mean': np.random.normal(1000, 200, n_anomaly), # Larger packets
            'Label': ['ATTACK'] * n_anomaly
        }
        
        # Combine data
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
        
        df = pd.DataFrame(all_data)
        df.to_csv('CICIDS2017_sample.csv', index=False)
        print("Sample data created and saved as 'CICIDS2017_sample.csv'")
    
    # Prepare features and labels
    feature_columns = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                      'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean']
    
    X = df[feature_columns].values
    
    # Encode labels: BENIGN -> 0, others -> 1
    le = LabelEncoder()
    y = le.fit_transform(df['Label'].values)
    y = (y > 0).astype(int)  # Convert to binary: 0 for BENIGN, 1 for attacks
    
    return X, y


def evaluate_model(model: QuantumAnomalyDetector, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and create visualizations."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Prediction Probabilities Distribution
    axes[0, 1].hist(y_pred_proba[y_test == 0], alpha=0.7, label='Normal', bins=30)
    axes[0, 1].hist(y_pred_proba[y_test == 1], alpha=0.7, label='Anomaly', bins=30)
    axes[0, 1].set_xlabel('Prediction Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Probability Distribution')
    axes[0, 1].legend()
    
    # PCA Visualization of Test Data
    X_test_processed, _ = model.preprocess_data(X_test, fit=False)
    scatter = axes[1, 0].scatter(X_test_processed[:, 0], X_test_processed[:, 1], 
                                c=y_test, cmap='coolwarm', alpha=0.7)
    axes[1, 0].set_xlabel('First Principal Component')
    axes[1, 0].set_ylabel('Second Principal Component')
    axes[1, 0].set_title('PCA - True Labels')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # PCA Visualization of Predictions
    scatter = axes[1, 1].scatter(X_test_processed[:, 0], X_test_processed[:, 1], 
                                c=y_pred, cmap='coolwarm', alpha=0.7)
    axes[1, 1].set_xlabel('First Principal Component')
    axes[1, 1].set_ylabel('Second Principal Component')
    axes[1, 1].set_title('PCA - Predicted Labels')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return accuracy, y_pred, y_pred_proba


def main():
    """Main function to run the quantum anomaly detection experiment."""
    print("🔬 Quantum Machine Learning Anomaly Detection for Network Security")
    print("=" * 70)
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    X, y = load_and_prepare_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Normal samples: {np.sum(y == 0)}, Anomaly samples: {np.sum(y == 1)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize quantum model
    print("\n🔮 Initializing Quantum Anomaly Detector...")
    qad = QuantumAnomalyDetector(n_qubits=4, n_layers=3)
    
    # Train the model
    print("\n🚀 Training quantum model...")
    history = qad.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=16
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate the model
    print("\n📈 Evaluating model performance...")
    accuracy, y_pred, y_pred_proba = evaluate_model(qad, X_test, y_test)
    
    print(f"\n✅ Final Test Accuracy: {accuracy:.4f}")
    print("\n🎯 Quantum anomaly detection completed successfully!")
    
    return qad, history, accuracy


if __name__ == "__main__":
    # Run the quantum anomaly detection experiment
    model, training_history, final_accuracy = main()