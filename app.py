import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Graph(id='graph',
              figure=dict(
                  layout=dict(
                      width = 800,
                      height = 500,
                      yaxis=dict(
                          scaleanchor="x",
                          scaleratio=1,
                      )
                  )
              )),
    html.Div(id='output-data-upload'),
])


def remove_empty(data_frame):
    null_columns = data_frame.isnull().all(axis=0)
    null_rows = data_frame.isnull().all(axis=1)

    return data_frame.loc[data_frame.index[~null_rows], data_frame.columns[~null_columns]]


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), skiprows=5)
            df = remove_empty(df)
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.Div([
            dcc.Input(
                id='editing-columns-name',
                placeholder='Enter a column name...',
                value='',
                style={'padding': 10}
            ),
            html.Button('Add Column', id='editing-columns-button', n_clicks=0),
            html.Button('Add Row', id='editing-rows-button', n_clicks=0)
        ], style={'height': 50}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": str(i), "id": str(i)} for i in df.columns],
            id='data-table',
            row_deletable=True
        ),
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    Output('data-table', 'columns'),
    [Input('editing-columns-button', 'n_clicks')],
    [State('editing-columns-name', 'value'),
     State('data-table', 'columns')])
def update_columns(n_clicks, value, existing_columns):
    if n_clicks > 0:
        existing_columns.append({
            'id': value, 'name': value,
            'renamable': True, 'deletable': True
        })
    return existing_columns


@app.callback(
    Output('data-table', 'data'),
    [Input('editing-rows-button', 'n_clicks')],
    [State('data-table', 'data'),
     State('data-table', 'columns')])
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


if __name__ == '__main__':
    app.run_server(debug=True)
