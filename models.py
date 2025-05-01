import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# MODELO SIMPLE 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob=0.2):
        """
        Modelo LSTM con Dropout.
        
        Args:
            input_size (int): Número de características de entrada.
            hidden_size (int): Número de unidades ocultas.
            output_size (int): Número de características de salida (empresas).
            num_layers (int): Número de capas LSTM.
            dropout_prob (float): Probabilidad de Dropout.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capa LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Capa totalmente conectada (Fully Connected)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        
        # LSTM Forward Pass
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)
        
        # Seleccionar la última salida de la secuencia
        out = out[:, -1, :]    # (batch_size, hidden_size)
        
        # Aplicar Dropout
        out = self.dropout(out)
        
        # Pasar por la capa fully connected
        out = self.fc(out)     # (batch_size, output_size)
        
        return out

## MODELO MULTIOUTPUT 

class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_ramas, empresas_por_rama, num_layers=1, dropout_prob=0.2):
        super(MultiOutputLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Capa compartida: LSTM
        self.shared_lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        
        # Ramas para cada grupo de empresas
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, empresas_por_rama)
            )
            for _ in range(n_ramas)
        ])
    
    def forward(self, x):
        # Capa compartida
        shared_out, _ = self.shared_lstm(x)  # (batch_size, seq_length, hidden_size)

        # Salidas individuales (una para cada rama)
        outputs = []
        for head in self.heads:
            head_out, _ = head[1](shared_out)  # Pasar por LSTM específica
            head_out = head_out[:, -1, :]     # Seleccionar la última salida
            outputs.append(head[2](head_out))  # Pasar por la capa de salida

        return outputs




class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # RNN Forward Pass
        out, _ = self.rnn(x)   # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]    # Tomar la última salida (batch_size, hidden_size)
        out = self.fc(out)     # Pasar por la capa fully connected (batch_size, output_size)
        return out


from utils import mean_positive_error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

def calculate_metrics(predictions, actuals):
    """
    Calcular RMSE, MPE, MAPE y R².
    
    Args:
        predictions (ndarray): Predicciones del modelo.
        actuals (ndarray): Valores reales.
    
    Returns:
        tuple: RMSE, MPE, MAPE y R².
    """
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions)
    mpe = mean_positive_error(actuals,predictions)
    r2 = r2_score(actuals, predictions)
    
    return rmse, mpe, mape, r2




def train_and_evaluate_model(model, train_loader, test_loader, num_epochs, learning_rate, weight_decay=0.0, optimizer_choice="adam", criterion='MSE', patience=5):
    """
    Entrenar y evaluar un modelo en PyTorch con Early Stopping.
    
    Args:
        model (nn.Module): Modelo de PyTorch.
        train_loader (DataLoader): DataLoader para el conjunto de entrenamiento.
        test_loader (DataLoader): DataLoader para el conjunto de prueba.
        num_epochs (int): Número de épocas para entrenar.
        learning_rate (float): Tasa de aprendizaje.
        weight_decay (float): Decaimiento de los pesos para regularización.
        optimizer_choice (str): Tipo de optimizador a usar ('adam', 'sgd', etc.).
        criterion (str): Tipo de función de pérdida ('MSE', 'Huber').
        patience (int): Número de épocas sin mejora en valid_loss antes de detener el entrenamiento.
    
    Returns:
        model: Modelo entrenado.
        valid_actuals: Valores reales del conjunto de validación.
        valid_preds: Predicciones del modelo para el conjunto de validación.
        metrics: Diccionario con métricas de entrenamiento y validación.
    """
    model = model.to(device)

    # Configurar pérdida y optimizador
    if criterion == 'MSE':
        criterion = nn.MSELoss()
    elif criterion == 'Huber': 
        criterion = torch.nn.HuberLoss()

    # Configurar optimizador
    if optimizer_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizador '{optimizer_choice}' no soportado.")

    # Almacenar métricas
    metrics = {
        "train_losses": [],
        "valid_losses": [],
        "train_rmse": [],
        "train_mpe": [],
        "train_mape": [],
        "train_r2": [],
        "valid_rmse": [],
        "valid_mpe": [],
        "valid_mape": [],
        "valid_r2": [],
    }

    # Early Stopping variables
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()  # Modo de entrenamiento
        train_loss = 0.0
        train_actuals, train_preds = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Guardar predicciones y valores reales
            train_preds.append(outputs.detach().cpu().numpy())
            train_actuals.append(y_batch.cpu().numpy())

        # Calcular métricas para entrenamiento
        train_preds = np.concatenate(train_preds, axis=0)
        train_actuals = np.concatenate(train_actuals, axis=0)
        train_rmse, train_mpe, train_mape, train_r2 = calculate_metrics(train_preds, train_actuals)

        metrics["train_losses"].append(train_loss / len(train_loader))
        metrics["train_rmse"].append(train_rmse)
        metrics["train_mpe"].append(train_mpe)
        metrics["train_mape"].append(train_mape)
        metrics["train_r2"].append(train_r2)

        # Evaluar en conjunto de prueba
        model.eval()  # Modo de evaluación
        valid_loss = 0.0
        valid_actuals, valid_preds = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item()

                # Guardar predicciones y valores reales
                valid_preds.append(outputs.cpu().numpy())
                valid_actuals.append(y_batch.cpu().numpy())
        # Calcular métricas para validación
        valid_preds = np.concatenate(valid_preds, axis=0)
        valid_actuals = np.concatenate(valid_actuals, axis=0)
        valid_loss = valid_loss / len(test_loader)
        valid_rmse, valid_mpe, valid_mape, valid_r2 = calculate_metrics(valid_preds, valid_actuals)

        metrics["valid_losses"].append(valid_loss)
        metrics["valid_rmse"].append(valid_rmse)
        metrics["valid_mpe"].append(valid_mpe)
        metrics["valid_mape"].append(valid_mape)
        metrics["valid_r2"].append(valid_r2)

        # Imprimir resultados de la época
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {metrics['train_losses'][-1]:.4f}, "
              f"Valid Loss: {metrics['valid_losses'][-1]:.4f}, Train RMSE: {train_rmse:.4f}, "
              f"Valid RMSE: {valid_rmse:.4f}, Train MAPE: {train_mape:.4f}, Valid MAPE: {valid_mape:.4f}, "
              f"Train MPE: {train_mpe:.4f}, Valid MPE: {valid_mpe:.4f}, Train R²: {train_r2:.4f}, Valid R²: {valid_r2:.4f}")

        # Early Stopping logic
        if valid_rmse < best_valid_loss:
            best_valid_loss = valid_rmse
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Guardar el mejor estado del modelo
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Restaurar el mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, valid_actuals, valid_preds, metrics


