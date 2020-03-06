import base64
import io
import redis

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import plotly.utils

import json

import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config.suppress_callback_exceptions = True

redis_instance = redis.StrictRedis.from_url("redis://localhost:6379", decode_responses=True)

options = [
    {'label': key, 'value': key}
    for key in redis_instance.keys()]



app.layout = html.Div([
    dcc.Dropdown(id="dropdown", options=options),
    dcc.Input(id='input', value=''),
    html.Button('Add Session', id='submit', n_clicks=0),
    html.Div([], id='layout-view'),
])


@app.callback(Output('dropdown', 'options'), [Input('submit', 'n_clicks')], [State('input', 'value'), State('dropdown', 'options')])
def callback(n_clicks, new_value, current_options):
    if n_clicks < 1 or not new_value:
        return current_options

    current_options.append({'label': new_value, 'value': new_value})
    redis_instance.append(new_value, json.dumps(html.Div([new_value]), cls=plotly.utils.PlotlyJSONEncoder))
    return current_options


@app.callback(Output('layout-view', 'children'), [Input('dropdown', 'value')])
def set_layout(value):
    if value:
        return json.loads(redis_instance.get(value))
    return []


if __name__ == '__main__':
    app.run_server(debug=True)