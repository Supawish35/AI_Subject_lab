import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class ANNClassifier:
    """
    Artificial Neural Network (ANN) Multi-Layer Perceptron Classifier.
    Encapsulates training, inference, feature normalization, detailed layer-by-layer
    forward propagation, and parameter export for interactive visualization.
    """
    def __init__(self, hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=600, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=False
        )
        self.classes_ = None
        self.feature_names = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        """Train the ANN model with automatic standard scaling."""
        X_mat = np.array(X, dtype=float)
        y_vec = np.array(y)

        # Fit feature standardizer
        X_scaled = self.scaler.fit_transform(X_mat)

        # Train Multi-Layer Perceptron
        self.model.fit(X_scaled, y_vec)
        self.classes_ = list(self.model.classes_)
        self.feature_names = list(feature_names) if feature_names is not None else [f"f_{i}" for i in range(X.shape[1])]
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input matrix."""
        X_mat = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X_mat)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution."""
        X_mat = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X_mat)
        return self.model.predict_proba(X_scaled)

    def forward_propagation_detailed(self, raw_input_vector: np.ndarray) -> dict:
        """
        Compute step-by-step activations through all layers of the neural network:
        Input Layer -> Hidden Layers -> Softmax Output Layer.
        Returns activation values for every individual neuron to enable live visualization.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        vec = np.array(raw_input_vector, dtype=float).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)[0]

        # Activations storage
        layers_activations = [vec_scaled.tolist()]
        current_a = vec_scaled

        for l_idx, (weights, bias) in enumerate(zip(self.model.coefs_, self.model.intercepts_)):
            # z = a * W + b
            z = np.dot(current_a, weights) + bias
            is_last_layer = (l_idx == len(self.model.coefs_) - 1)

            if is_last_layer:
                # Softmax activation for multi-class probability output
                exp_z = np.exp(z - np.max(z))
                current_a = exp_z / np.sum(exp_z)
            else:
                # Hidden layer activation (ReLU or Sigmoid)
                if self.activation == 'relu':
                    current_a = np.maximum(0, z)
                elif self.activation == 'logistic':
                    current_a = 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
                else:
                    current_a = np.tanh(z)

            layers_activations.append(current_a.tolist())

        probabilities = layers_activations[-1]
        best_idx = int(np.argmax(probabilities))
        predicted_class = self.classes_[best_idx]
        confidence = float(probabilities[best_idx])

        class_prob_map = {self.classes_[i]: float(probabilities[i]) for i in range(len(self.classes_))}

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': class_prob_map,
            'layer_activations': layers_activations,
            'input_scaled': vec_scaled.tolist()
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate performance metrics on a holdout test set."""
        X_mat = np.array(X_test, dtype=float)
        y_vec = np.array(y_test)
        y_pred = self.predict(X_mat)

        cm = confusion_matrix(y_vec, y_pred, labels=self.classes_)
        acc = accuracy_score(y_vec, y_pred)
        prec_w = precision_score(y_vec, y_pred, average='weighted', zero_division=0)
        rec_w = recall_score(y_vec, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_vec, y_pred, average='weighted', zero_division=0)

        prec_m = precision_score(y_vec, y_pred, average='macro', zero_division=0)
        rec_m = recall_score(y_vec, y_pred, average='macro', zero_division=0)
        f1_m = f1_score(y_vec, y_pred, average='macro', zero_division=0)

        prec_per = precision_score(y_vec, y_pred, labels=self.classes_, average=None, zero_division=0)
        rec_per = recall_score(y_vec, y_pred, labels=self.classes_, average=None, zero_division=0)
        f1_per = f1_score(y_vec, y_pred, labels=self.classes_, average=None, zero_division=0)

        per_class_list = []
        for idx, c in enumerate(self.classes_):
            per_class_list.append({
                'class': c,
                'precision': float(prec_per[idx] * 100),
                'recall': float(rec_per[idx] * 100),
                'f1': float(f1_per[idx] * 100),
                'support': int(np.sum(y_vec == c))
            })

        return {
            'accuracy': float(acc * 100),
            'precision_weighted': float(prec_w * 100),
            'recall_weighted': float(rec_w * 100),
            'f1_weighted': float(f1_w * 100),
            'precision_macro': float(prec_m * 100),
            'recall_macro': float(rec_m * 100),
            'f1_macro': float(f1_m * 100),
            'confusion_matrix': cm.tolist(),
            'per_class': per_class_list,
            'classes': self.classes_
        }

    def export_network_json(self) -> dict:
        """Export trained network architecture, weights, and scaler for browser inference."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        weights_export = [w.tolist() for w in self.model.coefs_]
        biases_export = [b.tolist() for b in self.model.intercepts_]

        return {
            'classes': self.classes_,
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'weights': weights_export,
            'biases': biases_export,
            'hidden_layer_sizes': list(self.hidden_layer_sizes),
            'activation': self.activation
        }
