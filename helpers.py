import cirq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class QuantumCircuitVisualizer:
    """Utility class for visualizing quantum circuits and their properties."""
    
    @staticmethod
    def visualize_circuit(circuit: cirq.Circuit, title: str = "Quantum Circuit"):
        """Visualize a Cirq circuit."""
        print(f"\n{title}")
        print("=" * len(title))
        print(circuit)
        
    @staticmethod
    def plot_circuit_depth_analysis(circuits: List[cirq.Circuit], labels: List[str]):
        """Analyze and plot circuit depths."""
        depths = [len(circuit) for circuit in circuits]
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, depths, color=['blue', 'green', 'red'][:len(labels)])
        plt.title('Quantum Circuit Depth Analysis')
        plt.xlabel('Circuit Type')
        plt.ylabel('Circuit Depth')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return depths

class QuantumFeatureMap:
    """Advanced quantum feature mapping strategies."""
    
    @staticmethod
    def angle_encoding(data: np.ndarray, qubits: List[cirq.Qubit]) -> cirq.Circuit:
        """Encode classical data using angle encoding."""
        circuit = cirq.Circuit()
        n_features = min(len(data), len(qubits))
        
        for i in range(n_features):
            # Normalize to [0, π] range
            angle = np.arctan(data[i]) + np.pi/2
            circuit.append(cirq.ry(angle)(qubits[i]))
            
        return circuit
    
    @staticmethod
    def amplitude_encoding(data: np.ndarray, qubits: List[cirq.Qubit]) -> cirq.Circuit:
        """Encode classical data using amplitude encoding."""
        circuit = cirq.Circuit()
        n_qubits = len(qubits)
        
        # Normalize data to create valid quantum state
        normalized_data = data / np.linalg.norm(data)
        
        # Pad or truncate to match 2^n_qubits
        target_size = 2 ** n_qubits
        if len(normalized_data) < target_size:
            normalized_data = np.pad(normalized_data, (0, target_size - len(normalized_data)))
        else:
            normalized_data = normalized_data[:target_size]
        
        # Use state preparation (simplified approach)
        # In practice, you'd use a more sophisticated state preparation algorithm
        for i, qubit in enumerate(qubits):
            if i < len(data):
                angle = 2 * np.arcsin(np.sqrt(abs(normalized_data[i])))
                circuit.append(cirq.ry(angle)(qubit))
        
        return circuit
    
    @staticmethod
    def basis_encoding(data: np.ndarray, qubits: List[cirq.Qubit]) -> cirq.Circuit:
        """Encode classical data using basis encoding."""
        circuit = cirq.Circuit()
        
        # Convert to binary representation
        for i, qubit in enumerate(qubits):
            if i < len(data) and data[i] > 0.5:  # Threshold at 0.5
                circuit.append(cirq.X(qubit))
                
        return circuit

class QuantumAnsatz:
    """Collection of quantum ansatz (parameterized circuits) for VQC."""
    
    @staticmethod
    def hardware_efficient_ansatz(qubits: List[cirq.Qubit], n_layers: int) -> Tuple[cirq.Circuit, List[cirq.Symbol]]:
        """Hardware-efficient ansatz with alternating single-qubit rotations and entangling gates."""
        circuit = cirq.Circuit()
        symbols = []
        
        for layer in range(n_layers):
            # Single-qubit rotations
            for i, qubit in enumerate(qubits):
                theta = cirq.Symbol(f'theta_{layer}_{i}')
                symbols.append(theta)
                circuit.append(cirq.ry(theta)(qubit))
            
            # Entangling gates
            for i in range(len(qubits) - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        
        return circuit, symbols
    
    @staticmethod
    def alternating_layered_ansatz(qubits: List[cirq.Qubit], n_layers: int) -> Tuple[cirq.Circuit, List[cirq.Symbol]]:
        """Alternating layered ansatz with RY and RZ rotations."""
        circuit = cirq.Circuit()
        symbols = []
        
        for layer in range(n_layers):
            # RY rotations
            for i, qubit in enumerate(qubits):
                theta_y = cirq.Symbol(f'theta_y_{layer}_{i}')
                symbols.append(theta_y)
                circuit.append(cirq.ry(theta_y)(qubit))
            
            # RZ rotations
            for i, qubit in enumerate(qubits):
                theta_z = cirq.Symbol(f'theta_z_{layer}_{i}')
                symbols.append(theta_z)
                circuit.append(cirq.rz(theta_z)(qubit))
            
            # Entangling layer
            if layer < n_layers - 1:
                for i in range(len(qubits) - 1):
                    circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
                # Circular entanglement
                if len(qubits) > 2:
                    circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
        
        return circuit, symbols

class QuantumMetrics:
    """Advanced metrics for quantum machine learning models."""
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "ROC Curve"):
        """Plot ROC curve for binary classification."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return roc_auc
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   title: str = "Precision-Recall Curve"):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return avg_precision
    
    @staticmethod
    def quantum_fidelity_analysis(model, X_test: np.ndarray, n_samples: int = 100):
        """Analyze quantum state fidelity for different inputs."""
        # Sample random test points
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        sample_data = X_test[indices]
        
        # This would require access to quantum states - simplified for demonstration
        print("Quantum Fidelity Analysis")
        print("=" * 30)
        print(f"Analyzed {n_samples} quantum states")
        print("Note: Full fidelity analysis requires quantum state access")
        
        return sample_data

class NetworkTrafficGenerator:
    """Generate realistic network traffic data for testing."""
    
    @staticmethod
    def generate_normal_traffic(n_samples: int = 500) -> Dict[str, np.ndarray]:
        """Generate normal network traffic patterns."""
        np.random.seed(42)
        
        return {
            'Flow Duration': np.random.exponential(1000, n_samples),
            'Total Fwd Packets': np.random.poisson(10, n_samples),
            'Total Backward Packets': np.random.poisson(8, n_samples),
            'Flow Bytes/s': np.random.exponential(5000, n_samples),
            'Flow Packets/s': np.random.exponential(20, n_samples),
            'Fwd Packet Length Mean': np.random.normal(500, 100, n_samples),
        }
    
    @staticmethod
    def generate_ddos_traffic(n_samples: int = 200) -> Dict[str, np.ndarray]:
        """Generate DDoS attack traffic patterns."""
        np.random.seed(123)
        
        return {
            'Flow Duration': np.random.exponential(100, n_samples),  # Short bursts
            'Total Fwd Packets': np.random.poisson(100, n_samples),  # Many packets
            'Total Backward Packets': np.random.poisson(1, n_samples),  # Few responses
            'Flow Bytes/s': np.random.exponential(100000, n_samples),  # High bandwidth
            'Flow Packets/s': np.random.exponential(200, n_samples),  # High packet rate
            'Fwd Packet Length Mean': np.random.normal(64, 10, n_samples),  # Small packets
        }
    
    @staticmethod
    def generate_port_scan_traffic(n_samples: int = 150) -> Dict[str, np.ndarray]:
        """Generate port scanning attack patterns."""
        np.random.seed(456)
        
        return {
            'Flow Duration': np.random.exponential(50, n_samples),  # Very short
            'Total Fwd Packets': np.random.poisson(1, n_samples),  # Single packets
            'Total Backward Packets': np.random.poisson(0, n_samples),  # No responses
            'Flow Bytes/s': np.random.exponential(100, n_samples),  # Low bandwidth
            'Flow Packets/s': np.random.exponential(10, n_samples),  # Low packet rate
            'Fwd Packet Length Mean': np.random.normal(40, 5, n_samples),  # Very small
        }
    
    @staticmethod
    def create_comprehensive_dataset(save_path: str = 'comprehensive_network_data.csv') -> pd.DataFrame:
        """Create a comprehensive network traffic dataset."""
        # Generate different types of traffic
        normal = NetworkTrafficGenerator.generate_normal_traffic(700)
        ddos = NetworkTrafficGenerator.generate_ddos_traffic(200)