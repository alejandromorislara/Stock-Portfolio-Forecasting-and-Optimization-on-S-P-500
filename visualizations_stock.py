import matplotlib.pyplot as plt
import numpy as np

import numpy as np

# def desescalar(actuals, preds, output_scale, output_mean):
#     # Asegurar que las predicciones y los valores reales tienen la forma correcta
#     actuals = np.array(actuals).reshape(-1, n_output_features)
#     preds = np.array(preds).reshape(-1, n_output_features)

#     # Desescalar
#     actuals_descaled = (actuals * output_scale) + output_mean
#     preds_descaled = (preds * output_scale) + output_mean

#     return actuals_descaled, preds_descaled


def plot_DL(actuals,preds,model,scaler):

    #     # Ahora scaler_y.scale_ y scaler_y.mean_ deberían tener forma (2,)
    #     # Aislar la escala y media de las salidas
    # n_output_features = actuals.shape[1]  # Número de salidas (2)
    # output_scale = scaler.scale_[:n_output_features]
    # output_mean = scaler.mean_[:n_output_features]

    # # Desescalar valores reales y predicciones
    # actuals_descaled, preds_descaled = desescalar(actuals, preds, output_scale, output_mean)

    # Graficar predicciones del modelo LSTM
    plt.plot(actuals[0], label='Actual')
    plt.plot(preds[0], label=f'{model} Prediction')
    
    plt.title(f"{model} Predictions vs Actuals")
    plt.legend()
    plt.show()
