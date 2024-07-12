import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

# Function to load and process data, train model, and make predictions
def load_and_predict(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']
    
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    
    for i in range(0, len(data)):
        new_data["Date"][i] = data['Date'][i]
        new_data["Close"][i] = data["Close"][i]
    
    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)
    dataset = new_data.values
    
    train = dataset[0:int(len(dataset)*0.8), :]
    valid = dataset[int(len(dataset)*0.8):, :]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    
    for i in range(60, len(train)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)
    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = lstm_model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    
    train = new_data[:int(len(dataset)*0.8)]
    valid = new_data[int(len(dataset)*0.8):]
    valid['Predictions'] = closing_price
    
    return train, valid

train_nse, valid_nse = load_and_predict("BTC-USD.csv")
train_stock1, valid_stock1 = load_and_predict("ETH-USD.csv")
train_stock2, valid_stock2 = load_and_predict("ADA-USD.csv")

app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='BTC-USD', children=[
            html.Div([
                html.H2("Actual vs Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="actual-data-nse",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train_nse.index,
                                y=train_nse["Close"],
                                mode='lines',
                                name='Train'
                            ),
                            go.Scatter(
                                x=valid_nse.index,
                                y=valid_nse["Close"],
                                mode='lines',
                                name='Actual'
                            ),
                            go.Scatter(
                                x=valid_nse.index,
                                y=valid_nse["Predictions"],
                                mode='lines',
                                name='Predicted Close'
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                
            ])
        ]),
        dcc.Tab(label='ETH-USD', children=[
            html.Div([
                html.H2("Actual vs Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="actual-data-stock1",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train_stock1.index,
                                y=train_stock1["Close"],
                                mode='lines',
                                name='Train'
                            ),
                             go.Scatter(
                                x=valid_stock1.index,
                                y=valid_stock1["Close"],
                                mode='lines',
                                name='Actual'
                            ),
                             go.Scatter(
                                x=valid_stock1.index,
                                y=valid_stock1["Predictions"],
                                mode='lines',
                                name='Predicted Close'
                            ),
                        ],
                        "layout": go.Layout(
                            title='Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
            ])
        ]),
        dcc.Tab(label='ADA-USD', children=[
            html.Div([
                html.H2("Actual vs Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="actual-data-stock2",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train_stock2.index,
                                y=train_stock2["Close"],
                                mode='lines',
                                name='Train'
                            ),
                            go.Scatter(
                                x=valid_stock2.index,
                                y=valid_stock2["Close"],
                                mode='lines',
                                name='Actual'
                            ),
                            go.Scatter(
                                x=valid_stock2.index,
                                y=valid_stock2["Predictions"],
                                mode='lines',
                                name='Predicted Close'
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
               
            ])
        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
