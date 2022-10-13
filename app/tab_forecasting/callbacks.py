from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from config.common import QUERIES, POSTGRES_URL

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Stock Forecasting'),
    dcc.Graph(id="time-series-chart"),
    html.P("Select stock:"),
    dcc.Dropdown(
        id="ticker",
        options=["ABBV", "AMGN", "ABC"],
        value="AMZN",
        clearable=False,
    ),
])


@app.callback(
    Output("time-series-chart", "figure"),
    Input("ticker", "value"))
def display_time_series(ticker):
    q = QUERIES['time_series'].replace('SYMBOL', ticker)
    df = pd.read_sql(q, POSTGRES_URL)
    print(df.loc[df.duplicated(subset=['date'])])
    fig = px.line(df, x='date', y='close', title=f'{ticker.upper()} Forecast')
    return fig


app.run_server(debug=True)
