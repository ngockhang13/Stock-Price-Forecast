# run: streamlit run arima_lstm.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
g_scaler = MinMaxScaler(feature_range=(10e-5,1))
g_scaler2 = MinMaxScaler(feature_range=(10e-5,1))
g_scaler_arima_lstm = MinMaxScaler(feature_range=(10e-5,1))
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import tensorflow as tf
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

plt.rcParams['lines.linewidth'] = 1
dpi = 1000
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['figure.figsize'] = 8, 8

#
g_column_select = 0
arima_max_p = 3
arima_max_q = 3
arima_m = 1
isSelectFile = False

g_Number_Epochs = 20
g_Number_Layer = 6
g_Number_Units = 128
g_data_traine_rate = 0.7
g_stock_data = 0
g_Drop = 0.3

g_model_arima  = None
g_model_LSTM = None
g_alpha = 0


st.set_page_config(
    page_title="Stock Price Prediction", page_icon="📊", initial_sidebar_state="expanded"
)

import datetime
first_time  = None

def deltaTime(isStart):
    global first_time
    if isStart:
        first_time = datetime.datetime.now()
        return None
    else:
        later_time = datetime.datetime.now()
        duration  = later_time - first_time
        duration_in_s = duration.total_seconds()
        return duration_in_s 


def modelArima(train_data):
    data_log = np.log(train_data[g_column_select])
    model_autoARIMA = auto_arima(data_log, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=arima_max_p, max_q=arima_max_q, # maximum p and q
                      m=arima_m,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

    return model_autoARIMA

def plot(data):
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=data["DATE"], y=df[g_column_select], name=g_column_select))
    st.plotly_chart(fig)
def plot2(Train, Test):
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=Train["DATE"], y=Train[g_column_select], name=g_column_select+ ' Train'))
    fig = fig.add_trace(go.Scatter(x=Test["DATE"], y=Test[g_column_select], name=g_column_select + ' Test'))
    st.plotly_chart(fig)   
def plot3(Train, Test, Pre):
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=Train["DATE"], y=Train[g_column_select], name=g_column_select+ ' Train'))
    fig = fig.add_trace(go.Scatter(x=Test["DATE"], y=Test[g_column_select], name=g_column_select + ' Test'))
    fig = fig.add_trace(go.Scatter(x=Test["DATE"], y=Pre, name=g_column_select + ' Predict'))
    st.plotly_chart(fig)

def initialModelLSTM(x_train):
    # min 3 layer
    model = None
    model = Sequential()
    model.add(LSTM(units = g_Number_Units, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(g_Drop))

    for i in range(g_Number_Layer - 3):
        model.add(LSTM(units = g_Number_Units, activation = 'relu', return_sequences = True))
        model.add(Dropout(g_Drop))

    model.add(LSTM(units = g_Number_Units, activation = 'relu'))
    model.add(Dropout(g_Drop))

    model.add(Dense(units = 1))
    return model

def plotTrainLSTM(dataTrain, dataTest, lable1, lable2):
    fig = go.Figure()
    X = np.array(range(len(dataTrain)))
    fig = fig.add_trace(go.Scatter(x= X, y=dataTrain, name=lable1))
    fig = fig.add_trace(go.Scatter(x=X, y=dataTest, name=lable2))
    st.plotly_chart(fig)
def plot2grap(data1, data2, lable1, lable2):
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x= np.array(range(len(data1))), y=data1, name=lable1))
    fig = fig.add_trace(go.Scatter(x=np.array(range(len(data2))), y=data2, name=lable2))
    fig.update_xaxes(nticks=10)
    st.plotly_chart(fig)
def plot3grap(data1, data2, data3, lable1, lable2, lable3):
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x= np.array(range(len(data1))), y=data1, name=lable1))
    fig = fig.add_trace(go.Scatter(x=np.array(range(len(data1), len(data1)+ len(data2))), y=data2, name=lable2))
    fig = fig.add_trace(go.Scatter(x=np.array(range(len(data1) , len(data1) + len(data3))), y=data3, name=lable3))
    st.plotly_chart(fig)
def plotWithName(data, name):
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=np.array(range(1, len(data) + 1)), y=data, name=name))
    st.plotly_chart(fig, use_container_width=True)
def process():
    global g_column_select, g_scaler, g_stock_data, g_scaler2, g_model_arima, g_model_LSTM, g_alpha
    df = g_stock_data
    train_data, test_data = df[3:int(len(df)*g_data_traine_rate)], df[int(len(df)*g_data_traine_rate):]
    placeholder = st.empty()
    placeholder.title("Stock Price Prediction")
    print("submit")
   
    st.markdown("### Data input:")
    plot(df)
    st.markdown("#### Split data into train and training set:")
    plot2(train_data, test_data)

    #---------------------------------------------------#
    st.markdown("# ARIMA Model")
    deltaTime(True)
    fitted = modelArima(train_data)
    t_ARIMA_Train = deltaTime(False)
    g_model_arima = filter
    # Forecast
    print("len: ", len(test_data[g_column_select]))
    deltaTime(True)
    fc= fitted.predict(n_periods = len(test_data[g_column_select]))
    t_ARIMA_Test = deltaTime(False)
    print(fc.shape)
    fc_series = pd.Series(fc, index=test_data.index)

    st.markdown("#### Plot data Predict")
    plot2grap(np.array(test_data[g_column_select]).reshape(-1), np.exp(fc_series), "Real", "ARIMA")
    st.markdown("#### Report performance")
    rmse = math.sqrt(mean_squared_error(np.log(test_data[g_column_select]), fc))
    st.markdown('**' +'RMSE: '+str(rmse)+ '**')
    mae = mean_absolute_error(np.log(test_data[g_column_select]), fc)
    st.markdown('**' +'MAE: '+str(mae)+ '**')
    
    #----------------------------------------------------------------------------------#
    st.markdown("# LSTM Model")
    data_training_array = g_scaler.fit_transform(np.array(train_data[g_column_select]).reshape(-1,1))
    data_test_array = g_scaler.fit_transform(np.array(test_data[g_column_select]).reshape(-1,1))

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    index_test = []
    for i in range(60, data_training_array.shape[0]):
        x_train.append(data_training_array[i-60: i])
        y_train.append(data_training_array[i, 0])

    for i in range(60, data_test_array.shape[0]):
        x_test.append(data_test_array[i-60: i])
        y_test.append(data_test_array[i, 0])
        index_test.append(i)
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    # run model
    with tf.device('/CPU:0'):
        deltaTime(True)
        modelLSTM = initialModelLSTM(x_train)
        modelLSTM.compile(optimizer = 'adam', loss = 'mean_squared_error')
        history  = modelLSTM.fit(x_train, y_train, epochs = g_Number_Epochs,
                validation_data = (x_test, y_test),shuffle= True, use_multiprocessing=True)
        t_LSTM_Train = deltaTime(False)        
        g_model_LSTM = modelLSTM
        # list all data in history
        st.markdown("#### Summarize history for loss")
        plotTrainLSTM(history.history['loss'], history.history['val_loss'],"Train", "Test")
        # Making predictions
        deltaTime(True)
        y_predicted = modelLSTM.predict(x_test)
        t_LSTM_Test = deltaTime(False) 
    y_predicted = g_scaler.inverse_transform(y_predicted.reshape(-1, 1) ) # * max_test
    y_test = g_scaler.inverse_transform(y_test.reshape(-1, 1) ) # * max_test
    y_predicted = np.array(y_predicted).reshape(-1)
    st.markdown("#### Plot data Predict")

    plot2grap(y_test.reshape(-1), y_predicted.reshape(-1), "Real", "LSTM")
    st.markdown("#### Report performance")
    y_test_log = np.log(y_test)
    y_predicted_log = np.log(y_predicted)

    rmse_l = math.sqrt(mean_squared_error(y_test_log, y_predicted_log))
    st.markdown('**' +'RMSE: '+str(rmse_l)+ '**')
    mae_l = mean_absolute_error(y_test_log, y_predicted_log)
    st.markdown('**' +'MAE: '+str(mae_l)+ '**')
    
    # --------------------------------------------------------------------------- #
    st.markdown("# ARIMA-LSTM")
    arima_predict_ = np.exp([fc[i] for i in index_test])
   
    data_training_rate = 0.7
    number_data_training = int(len(y_test) * data_training_rate)

    deltaAll = y_test.reshape(-1)- arima_predict_.reshape(-1)
    delta = deltaAll[0: number_data_training-1]
    # pre data for training LSTM
    data_training_array_lstm = g_scaler_arima_lstm.fit_transform(np.array(delta).reshape(-1,1))
    data_test_array_lstm = g_scaler_arima_lstm.fit_transform(np.array(deltaAll[number_data_training: -1]).reshape(-1,1))

    x_train_lstm = []
    y_train_lstm  = []
    x_test_lstm  = []
    y_test_lstm  = []
    index_test_lstm  = []
    for i in range(60, data_training_array_lstm.shape[0]):
        x_train_lstm.append(data_training_array_lstm[i-60: i])
        y_train_lstm.append(data_training_array_lstm[i, 0])

    for i in range(60, data_test_array_lstm.shape[0]):
        x_test_lstm.append(data_test_array_lstm[i-60: i])
        y_test_lstm.append(data_test_array_lstm[i, 0])
        index_test_lstm.append(i)
    
    x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)
    x_test_lstm, y_test_lstm = np.array(x_test_lstm), np.array(y_test_lstm)
    # run model
    with tf.device('/CPU:0'):
        deltaTime(True) 
        modelLSTM_2 = initialModelLSTM(x_train_lstm)
        modelLSTM_2.compile(optimizer = 'adam', loss = 'mean_squared_error')
        history  = modelLSTM_2.fit(x_train_lstm, y_train_lstm, epochs = g_Number_Epochs,
                validation_data = (x_test_lstm, y_test_lstm),shuffle= True, use_multiprocessing=True)
        t_hybrid_train = deltaTime(False) 
        # Making predictions
        deltaTime(True) 
        y_predicted_lstm = modelLSTM_2.predict(x_test_lstm)
        t_hybrid_test = deltaTime(False) 
    y_predicted_lstm = g_scaler_arima_lstm.inverse_transform(y_predicted_lstm.reshape(-1, 1) ) # * max_test

    y_predicted_lstm = np.array(y_predicted_lstm).reshape(-1)

    y_predicted_lstm_arima = arima_predict_.reshape(-1)[number_data_training +60 : -1] + y_predicted_lstm
    t_hybrid_test = deltaTime(False) 

    st.markdown("#### Plot data predict")
    plot2grap(y_test.reshape(-1)[number_data_training +60 : -1], y_predicted_lstm_arima, "Real", "ARIMA-LSTM")
    st.markdown("#### Report performance")
    y_predicted_lstm_arima_log = np.log(y_predicted_lstm_arima).reshape(-1)
    y_test_lstm_arima_log = y_test_log[number_data_training +60 : -1].reshape(-1)

    rmse_h_2 = math.sqrt(mean_squared_error(y_test_lstm_arima_log, y_predicted_lstm_arima_log))
    st.markdown('**' +'RMSE: '+str(rmse_h_2)+ '**')
    mae_h_2 = mean_absolute_error(y_test_lstm_arima_log, y_predicted_lstm_arima_log)
    st.markdown('**' +'MAE: '+str(mae_h_2)+ '**')
    
    #***********************************************************************************************#
    st.markdown("# ARIMA-LSTM cải tiến")
    ts = 0
    ms = 0
    deltaTime(True)
    for i in range(len(y_predicted)):
        ts = ts + (arima_predict_[i] - y_predicted[i])*(y_test[i] - y_predicted[i])
        ms = ms + (arima_predict_[i] - y_predicted[i])*(arima_predict_[i] - y_predicted[i])
    alpha = ts/ms
    
    if alpha[0] < 0:
        alpha[0] = 0
    elif alpha[0]>1:
        alpha[0] = 1
    t_hybrid2_train = deltaTime(False)     
    # --------------------------------------------------------------------------- #
    st.markdown('**' +'Alpha: '+str(alpha[0]) + '**')
    g_alpha = alpha[0]
    y_predicted_hybrid_2 = []
    deltaTime(True)
    for i in range(len(y_predicted)):
        y_predicted_hybrid_2.append(arima_predict_[i] * alpha[0] + y_predicted[i] * (1 -alpha[0]))
    t_hybrid2_test = deltaTime(False)
    
    st.markdown("#### Plot data predict")
    plot2grap(y_test.reshape(-1), y_predicted_hybrid_2, "Real", "ARIMA-LSTM cải tiến")
    st.markdown("#### Report performance")
    y_predicted_hybrid2_log = np.log(y_predicted_hybrid_2)
    rmse_h2 = math.sqrt(mean_squared_error(y_test_log, y_predicted_hybrid2_log))
    st.markdown('**' +'RMSE: '+str(rmse_h2)+ '**')
    mae_h2 = mean_absolute_error(y_test_log, y_predicted_hybrid2_log)
    st.markdown('**' +'MAE: '+str(mae_h2)+ '**')
    # --------------------------------------------------------------------------- #
    st.markdown("# Summary ")
    st.markdown("#### Report performance")
    table = {'Name': ["ARIMA" , "LSTM", "ARIMA-LSTM", "ARIMA-LSTM cải tiến"], 'Time(s)':[t_ARIMA_Train+ t_ARIMA_Test, t_LSTM_Train+ t_LSTM_Test, t_hybrid_train+ t_hybrid_test, t_hybrid2_train+ t_hybrid2_test],'RMSE': [rmse, rmse_l, rmse_h_2, rmse_h2], 'MAE': [mae, mae_l, mae_h_2, mae_h2]}
    st.table(table)
    #plot
    st.markdown("#### Plot data Predict")
    plot2grap(np.array(test_data[g_column_select]).reshape(-1), np.exp(fc_series), "Real", "ARIMA")
    plot2grap(y_test.reshape(-1), y_predicted.reshape(-1), "Real", "LSTM")
    plot2grap(y_test.reshape(-1)[number_data_training +60 : -1], y_predicted_lstm_arima, "Real", "ARIMA-LSTM")
    plot2grap(y_test.reshape(-1), y_predicted_hybrid_2, "Real", "ARIMA-LSTM cải tiến")


with st.sidebar:
    
    uploaded_file = st.file_uploader("Upload CSV", type=".csv")
    if uploaded_file:
        g_stock_data = pd.read_csv(uploaded_file)
        df = g_stock_data
        isSelectFile = True
        st.markdown("#### Data preview")
        st.dataframe(df.head())

        ab = st.selectbox("Feature", options=df.columns[1:])
        if ab:
            g_column_select = ab
            print("ab: ", ab)
        g_data_traine_rate = st.slider(
                "Training data rate",
                min_value=0.1,
                max_value=0.9,
                value=0.7,
                step=0.05
            )
        if g_data_traine_rate:
            print("g_data_traine_rate: ", g_data_traine_rate)
        st.sidebar.form("parameters")
            
        st.markdown("### Parameters ARIMA")
        arima_max_p = st.number_input(
            label="ARIMA_max_p",
            value=3,
            min_value = 0,
            format = "%d"
        )
        arima_max_q = st.number_input(
            label="ARIMA_max_q",
            value=3,
            min_value = 0,
            format = "%d"
        )
        arima_m = st.number_input(
            label="ARIMA_m",
            value=1,
            min_value = 0,
            format = "%d"
        )
        st.markdown("### Parameters LSTM")
        i_number_Epochs = st.number_input(
            label="Number Epochs", 
            value=20,
            format = "%d",
            min_value= 1
        )
        if i_number_Epochs:
            g_Number_Epochs = i_number_Epochs
            print("i_number_Epochs: ", i_number_Epochs)

        g_Number_Layer = st.number_input(
            label="Number of layers ",
            value=6,
            format = "%d",
            min_value= 3
        )    
        g_Number_Units = st.number_input(
            label="Number Units ",
            value=128,
            format = "%d",
            min_value= 2
        )
        g_Drop= st.slider(
            "Dropout",
            min_value=0.01,
            max_value=0.9,
            value=0.3,
            step=0.01
        )
        
        submit = st.button("Apply changes", on_click=process)