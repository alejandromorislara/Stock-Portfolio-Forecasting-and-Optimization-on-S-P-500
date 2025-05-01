import pandas as pd
import yfinance as yf
import os 
from fredapi import Fred
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# with open(r'C:\Users\user\OneDrive - Universidad de Oviedo\Escritorio\UNI\3ºAÑO\LAB_IACD\Proyecto_2_Lab_IACD\api_key.txt', 'r') as file:
#     api_key = file.read().strip()
# fred = Fred(api_key)

def extract_sp500_companies(url = "https://es.wikipedia.org/wiki/Anexo:Compa%C3%B1%C3%ADas_del_S%26P_500" , export = True) : 
    # Leer todas las tablas de la página
    tables = pd.read_html(url)
    df = tables[0]
    if export : 
        df.to_csv("data/sp500_companies.csv")
    return df


def extract_stock(stock_symbol, start_date = "2010-01-01", end_date = "2024-01-31" ) : 
    data = yf.download(stock_symbol, start = start_date, end = end_date)
    if data.empty:
        print(f"No data found for {stock_symbol}.")
        return None
    return data

def process_data(raw_data):
    """
    Procesa datos de acciones eliminando las primeras líneas no relevantes
    y configurando las columnas adecuadas.

    Args:
        raw_data (str): Ruta al archivo de datos bruto (CSV o similar).

    Returns:
        pd.DataFrame: Datos procesados con las columnas especificadas.
    """
    # Nombre de las columnas
    columns = ["Date", "AdjustedClose", "Close", "High", "Low", "Open", "Volume"]
    data = pd.read_csv(raw_data, skiprows=3, header=None)
    data.columns = columns
    data['Date'] = pd.to_datetime(data['Date'], format="ISO8601")
    
    return data

def join_stock_data(folder=r"data/stocks",export = True,axis=True):
    """
    Une los datos de acciones desde múltiples archivos CSV en un único DataFrame.
    
    Args:
        folder (str): Ruta al directorio que contiene los archivos CSV de datos de acciones.
        export (bool): Si se debe exportar el DataFrame combinado a un archivo CSV.
        axis (bool): Si se combinan los datos horizontalmente (True) o verticalmente (False).
        
    Returns:
        pd.DataFrame: DataFrame con todos los datos combinados.
    """
    all_data = []
    date_column = None  # Para guardar la columna 'Date' una vez

    for stock_data in os.listdir(folder):
        file_path = os.path.join(folder, stock_data)
        df = pd.read_csv(file_path)

        # Guardar solo la primera aparición de 'Date'

        symbol = os.path.splitext(stock_data)[0]

        if axis: 
             # Extraer la columna Date una sola vez
            if date_column is None:
                date_column = df[['Date']] 
            cols=['AdjustedClose', 'Close', 'High', 'Low', 'Open', 'Volume']
            df = df[['Date']+cols]
            df = df.set_index('Date')  # Usar 'Date' como índice para alineación

            df.columns=[f'{col}_{symbol}' for col in cols]
            # print(df.columns)
        else: 
            df["stock_symbol"] = symbol
           
            df = df[['Date', 'AdjustedClose', 'Close', 'High', 'Low', 'Open', 'Volume', 'stock_symbol']]
       
        all_data.append(df)
    if axis: 
            final_dataset = pd.concat(all_data,axis=1 ,ignore_index=False)
            final_dataset.reset_index(inplace=True)

            if export : 
                    final_dataset.to_csv(r'data/final_hztal.csv', index=False)
            return final_dataset
    else: 
            final_dataset = pd.concat(all_data, ignore_index=True)
            # print(final_dataset.head)
            # print(final_dataset.columns)
            if export : 
                final_dataset.to_csv(r'data/final_vtal.csv', index=False)
            return final_dataset

   

def extract_macroeconomics(series,start_date = "2010-01-01", end_date = "2024-01-31"):
    if series in ['^GSPC','^IXIC','^RUT','^STOXX50E','^FTSE','CL=F','SI=F','GC=F','^HSI','NG=F','ZC=F','EURUSD=X','BTC-USD','HO=F','ZC=F']:
            raw_macro_data=extract_stock(series,start_date,end_date)
            if raw_macro_data is not None : 
                raw_macro_data.to_csv(f"data/raw_macro/{series}.csv")
                macro_data = process_data(f"data/raw_macro/{series}.csv")
                macro_data=macro_data[['Date','Close']]
                macro_data.rename(columns={'Close':f'{series}'},inplace=True)
                
    else: 
            # Obtener la serie 
            macro_data = fred.get_series(series, observation_start=start_date, observation_end=end_date)
            macro_data=pd.DataFrame(macro_data,columns=[f'{series}'])
            macro_data.reset_index(names='Date',inplace=True)
            macro_data['Date']=pd.to_datetime(macro_data['Date'],format="ISO8601")
    data=macro_data
    macro_data.to_csv(f"data/macro/{series}.csv")


    return data

def join_macro(start,end):
     # DataFrame para almacenar todos los datos de entrenamiento
    series_ids = [
        'GDP',       # Producto Interno Bruto (PIB) de EE. UU.
        'UNRATE',    # Tasa de desempleo
        'CPIAUCSL',  # Índice de Precios al Consumidor (CPI) para todos los consumidores urbanos
        'PAYEMS',    # Nóminas no agrícolas (Nonfarm Payrolls)
        'FEDFUNDS',  # Tasa de fondos federales (Federal Funds Rate)
        'DGS10',     # Rendimiento del bono del Tesoro a 10 años
        'M1SL',      # Oferta monetaria M1
        'M2SL',      # Oferta monetaria M2
        '^GSPC',     # Índice S&P 500 (mercado bursátil de EE. UU.)
        'INDPRO',    # Índice de Producción Industrial
        'RSAFS',     # Ventas minoristas y servicios alimentarios
        'EXCAUS',    # Tipo de cambio del dólar estadounidense al dólar canadiense
        'BOPGSTB',   # Balanza comercial de bienes y servicios
        'GFDEBTN',   # Deuda del Gobierno Federal de EE. UU.
        'FGEXPND',   # Gasto total del gobierno federal
        'PCEPI',     # Índice de Precios de Gastos de Consumo Personal (PCE Price Index)
        'PPIACO',    # Índice de Precios al Productor (Producer Price Index)
        '^IXIC',     # NASDAQ (mercado bursátil)
        '^RUT',      # Índice Russell 2000 (empresas pequeñas de EE. UU.)
        '^STOXX50E', # Índice EURO STOXX 50 (mercado bursátil de Europa)
        '^FTSE',     # Índice FTSE 100 (mercado bursátil del Reino Unido)
        'CL=F',      # Futuros del petróleo crudo
        'SI=F',      # Futuros de la plata
        'GC=F',      # Futuros del oro
        '^HSI',      # Índice Hang Seng (mercado bursátil de Hong Kong)
        'NG=F',      # Futuros del gas natural
        'ZC=F',      # Futuros del maíz
        'EURUSD=X',  # Tipo de cambio del euro al dólar estadounidense
        'BTC-USD',   # Precio del Bitcoin en dólares estadounidenses
        'HO=F',      # Futuros del combustible de calefacción
        'ZC=F'       # Futuros del maíz (repetido)
    ]

    dataframes=[]
    for series in series_ids: 
        try:
            data=extract_macroeconomics(series,start,end)
        except Exception as error :
            print(f'Serie {series}  {error}')
        data['Date']=pd.to_datetime(data['Date']).dt.tz_localize(None) 
        dataframes.append(data)
    # Eliminar duplicados en cada DataFrame antes de concatenar
    dataframes = [df.drop_duplicates(subset='Date') for df in dataframes]
    final_df = pd.concat(dataframes, ignore_index=True)
    data=final_df
    date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max())

    # Crear un DataFrame con el rango de fechas único
    unique_dates = pd.DataFrame({'Date': date_range})

    # Agrupar por fecha consolidando los valores en una fila por fecha
    data_grouped = data.groupby('Date').first().reset_index()

    # Realizar un merge con el rango de fechas para asegurar que todas las fechas estén presentes
    final_df = pd.merge(unique_dates, data_grouped, on='Date', how='left')

    # Ordenar por fecha si es necesario
    final_df.sort_values(by='Date', inplace=True)
    # 
    # # Aplicar backfill a nivel de columnas
    final_df.bfill( inplace=True)
    final_df.ffill(inplace=True)

    # Mostrar resultado
    final_df.to_csv(f"data/macro_data.csv")
    return final_df

def get_technical_indicators(group, symbol, vertical=True):
    '''
    Genera indicadores técnicos sin incluir columnas originales como Close, High, Low, etc.
    '''
    df = pd.DataFrame()

    if vertical:
        symbol = group['stock_symbol'].unique()

    group = group.sort_values(by='Date')  # Asegúrate de que los datos estén ordenados por fecha

    # Generar solo indicadores técnicos
    df[f'MACD_{symbol}'] = ta.trend.macd(group['Close'])
    df[f'CCI_{symbol}'] = ta.trend.cci(group['High'], group['Low'], group['Close'], window=20)
    df[f'ATR_{symbol}'] = ta.volatility.average_true_range(group['High'], group['Low'], group['Close'], window=14)
    df[f'BOLL_upper_{symbol}'] = ta.volatility.bollinger_hband(group['Close'], window=20)
    df[f'BOLL_lower_{symbol}'] = ta.volatility.bollinger_lband(group['Close'], window=20)
    df[f'EMA20_{symbol}'] = ta.trend.ema_indicator(group['Close'], window=20)
    df[f'MA5_{symbol}'] = group['Close'].rolling(window=5).mean()
    df[f'MA10_{symbol}'] = group['Close'].rolling(window=10).mean()
    df[f'MTM6_{symbol}'] = group['Close'].pct_change(periods=6)
    df[f'MTM12_{symbol}'] = group['Close'].pct_change(periods=12)
    df[f'ROC_{symbol}'] = ta.momentum.roc(group['Close'], window=12)
    df[f'SMI_{symbol}'] = ta.momentum.stoch_signal(group['High'], group['Low'], group['Close'], window=14, smooth_window=3)
    df[f'WVAD_{symbol}'] = ((group['Close'] - group['Open']) / (group['High'] - group['Low']) * group['Volume']).fillna(0)
    df[f'RSI_{symbol}'] = ta.momentum.rsi(group['Close'], window=20)

    # Rellenar datos faltantes
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

def join_technical_indicators(database,folder = r"data/stocks", export=True, axis=True): 
    """
    Genera indicadores técnicos y une los datos en un único DataFrame, excluyendo columnas originales como Close, High, etc.
    
    Args:
        database (pd.DataFrame): DataFrame que contiene los datos de entrada.
        export (bool): Si se debe exportar el DataFrame combinado a un archivo CSV.
        axis (bool): Si se combinan los datos horizontalmente (True) o verticalmente (False).
        
    Returns:
        pd.DataFrame: DataFrame con indicadores técnicos.
    """
    dataframe = []
    
    date_file = None

    for stock_data in os.listdir(folder):
        symbol = os.path.splitext(stock_data)[0]
        columnas_excluidas=[col for col in database.columns if col.endswith(f"_{symbol}")]
        # Filtrar columnas relacionadas con el símbolo y mantener solo 'Date'
        cols = columnas_excluidas + ['Date']
        df = database[cols]
        
        # Cambiar los nombres de las columnas para eliminar el sufijo del símbolo
        df.columns = [col.rsplit('_', 1)[0] if '_' in col else col for col in df.columns]
        
        # Generar indicadores técnicos, excluyendo columnas originales como Close, High, Low, etc.
        resultados = get_technical_indicators(df, symbol, vertical=not(axis))

        # resultados_limpios=resultados.drop(columns=columnas_excluidas)
        if date_file is None: 
             date_file=database['Date']
             dataframe.append(date_file)
        dataframe.append(resultados)
    
    # Concatenar todos los resultados de indicadores técnicos
    final_df = pd.concat(dataframe, axis=1 if axis else 0)
    
    # Si se desea exportar, guardar el resultado
    if export:
        final_df.to_csv(f"data/combined_data.csv", index=False)
    
    return final_df




def preprocess_and_pca(X_train, X_test, target_keyword="Close", exclude_keyword="Adjusted", show_plot=True):
    """
    Realiza la preprocesamiento y PCA en un conjunto de datos.

    Args:
        X_train (DataFrame): Datos de entrenamiento.
        X_test (DataFrame): Datos de prueba.
        show_plot (bool): Si True, muestra un gráfico de varianza explicada acumulada.

    Returns:
        tuple: X_train_pca, X_test, varianza_acumulada
    """
    print("Inicio del preprocesamiento y PCA...")



    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Calcular PCA temporal para determinar componentes necesarios
    print("Hora antes del PCA:", datetime.now().time())
    pca_temp = PCA(n_components=0.975)
    pca_temp.fit(X_train)

    # Determinar componentes según el criterio de Kaiser
    autovalores = pca_temp.explained_variance_
    media_autovalores = np.mean(autovalores)

    criterio_kaiser_cov = autovalores > media_autovalores
    criterio_kaiser_cor = autovalores > 1
    print(f"Número de componentes según criterio de Kaiser (covarianza): {np.sum(criterio_kaiser_cov)}")
    print(f"Número de componentes según criterio de Kaiser (correlación): {np.sum(criterio_kaiser_cor)}")

    # Aplicar Incremental PCA con número óptimo de componentes
    ipca = IncrementalPCA(n_components=np.sum(criterio_kaiser_cor))
    # ipca = IncrementalPCA(n_components=3)

    X_train_pca = ipca.fit_transform(X_train)
    varianza_explicada = ipca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)

    print("Hora después del PCA:", datetime.now().time())

    # Mostrar gráfico si se solicita
    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='--')
        plt.title('Varianza Explicada Acumulada')
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.grid(True)
        plt.show()

    return X_train_pca, X_test, varianza_acumulada

def mean_positive_error(y_true, y_pred):
    """
    Calcula el Mean Positive Error (MPE).
    Args:
        y_true: Valores reales (numpy array o pandas DataFrame)
        y_pred: Valores predichos (numpy array o pandas DataFrame)
    Returns:
        Mean Positive Error (MPE)
    """
    errors = np.maximum(y_pred - y_true, 0)  # max(Y_hat - Y, 0)
    mpe = np.mean(errors)  # Promedio de los errores positivos
    return mpe

def transformar_a_tensor_3d(df, empresas, columnas_macro):
    """
    Convierte un DataFrame en un tensor 3D (Tiempo * Empresas * Características).
    """
    fechas = df['Date'].unique()
    num_fechas = len(fechas)
    num_empresas = len(empresas)
    num_caracteristicas_empresa = max(len([col for col in df.columns if col.endswith(f"_{empresa}")]) for empresa in empresas)
    
    num_caracteristicas_macro = len(columnas_macro)
    num_caracteristicas_total = num_caracteristicas_empresa + num_caracteristicas_macro

    print('Num Empresas',num_empresas)
    print('Num Fechas ', num_fechas)
    print('Num Características Empresa',num_caracteristicas_empresa )
    print('Num Características Macro',num_caracteristicas_macro)

    # Inicializar el tensor
    tensor = np.zeros((num_fechas, num_empresas, num_caracteristicas_total))
    


    # Llenar el tensor

    for t, fecha in enumerate(fechas):
        df_fecha = df[df['Date'] == fecha]
        # Variables macroeconómicas para la fecha actual
        macro_vals = df_fecha[columnas_macro].iloc[0].values if not df_fecha.empty else np.zeros(num_caracteristicas_macro)
        print('Fecha actual',fecha)

        for i, empresa in enumerate(empresas):
            # Columnas específicas de la empresa
            cols_empresa = sorted([col for col in df.columns if col.endswith(f"_{empresa}")])
            adjusted_close_index = [i for i, col in enumerate(cols_empresa) if "AdjustedClose" in col]
            cols_empresa = (
                [cols_empresa.pop(adjusted_close_index[0])] + cols_empresa
                if adjusted_close_index
                else cols_empresa
            )

            datos_empresa = df_fecha[cols_empresa].values.flatten() if not df_fecha.empty else np.zeros(len(cols_empresa))
            # Combinar datos de empresa y variables macroeconómicas
            datos_combinados = np.concatenate([datos_empresa, macro_vals])
            tensor[t, i, :] = datos_combinados
    
    return tensor


def crear_ventanas_temporales(tensor, ventana, horizonte):
    """
    Crear ventanas temporales utilizando únicamente la característica AdjustedClose.
    
    tensor: tensor 3D (tiempo, empresas, características).
    ventana: tamaño de la ventana de tiempo.
    horizonte: horizonte de predicción.
    
    Retorna:
    - X: tensor 3D de características para la ventana de tiempo.
    - y: matriz con el valor de AdjustedClose para el horizonte predicho.
    """
    X = []
    y = []
    for t in range(ventana, tensor.shape[0] - horizonte):
        # Extraer la ventana temporal
        X.append(tensor[t - ventana:t, :, :])  # Mantén todas las características en X
        # Seleccionar únicamente AdjustedClose para y (primer columna de características)
        y.append(tensor[t + horizonte, :, 0])  # Supone que AdjustedClose está en la primera posición
    return np.array(X), np.array(y)
