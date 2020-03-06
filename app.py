import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate

import pandas as pd

from regression import FourParametricLogistic

import numpy as np

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
            'width': 'calc(100% - 10px)',
            'height': '20px',
            'lineHeight': '20px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '5px'
        },
        multiple=False
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='graph',
                      figure=dict(
                          layout=dict(
                              yaxis=dict(
                                  scaleanchor="x",
                                  scaleratio=1,
                              )
                          )
                      )),
        ], style={'flex': '1'}),
        html.Div([
            html.Div([
                html.Label(
                    [
                        "Set concentrations",
                        dash_table.DataTable(
                            id='table-std-x',
                            columns=[{"id": "x", "name": "Concentration"}],
                            data=[
                                dict(x=0) for _ in range(8)
                            ],
                            editable=True,
                            row_deletable=True),
                        html.Button("Add x",
                                    id='button-add-x-std',
                                    n_clicks=0,
                                    style={'margin': "5px 5px 0 0"}),
                        html.Button(
                            "Update",
                            id="button-std-x-update",
                            n_clicks=0,
                            style={'margin': "5px 0 0 0"})
                    ]
                )
            ], style={'flex': '1', 'padding': '0 10px'}),
            html.Div(
                [
                    html.Label(
                        [
                            "Set Standard",
                            dcc.Dropdown(
                                id="dropdown-standard",
                                multi=True
                            ),
                            html.Button(
                                "Update standard",
                                id="button-std-y-update",
                                n_clicks=0,
                                style={'margin': "5px 0 0 0"}
                            ),
                        ]
                    ),
                    html.Label(
                        [
                            "Manage traces",
                            dcc.Dropdown(id="traces-dropdown")
                        ]
                    ),
                    html.Label(
                        [
                            "New trace",
                            dcc.Dropdown(id="trace-dropdown"),
                            dcc.Input(
                                id='trace-input',
                                placeholder="Enter new trace name",
                                style={'margin': "5px 5px 0 0", 'float': 'left'}),
                            html.Button("Add trace", style={'margin': "5px 0 0 5px"}),
                        ],
                    ),
                ], style={'flex': '2', 'padding': '0 10px'})
        ], style={'flex': '1', 'display': 'flex', 'overflowY': 'auto'})
    ], style={'display': 'flex', 'height': '400px'}),
    html.Div(id='output-data-upload'),
    dcc.Store(id='memory-std-xdata'),
    dcc.Store(id='memory-std-ydata'),
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
        html.Div(
            [
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": str(i), "id": str(i)} for i in df.columns],
                    id='table-data',
                    editable=True,
                    row_deletable=True
                ),
            ], style={'flex': '3'}
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P("Manage Data", style={'float: right'}),
                        html.Button(
                            'Add to Standard',
                            id='button-add-to-standard',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 5px 5px 0'}
                        ),
                        html.Button(
                            'Add to data',
                            id='button-add-to trace',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 5px 5px 0'}
                        )
                    ], className="container"
                ),
                html.Div(
                    [
                        html.P("Manage Table", style={'float: right'}),
                        dcc.Input(
                            id='editing-columns-name',
                            placeholder='Enter a column name...',
                            value='',
                            style={'float': 'right', 'margin': '0 5px 5px 0'}
                        ),
                        html.Button(
                            'Add Column',
                            id='editing-columns-button',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 5px 5px 0'}
                        ),
                        html.Button(
                            'Add Row',
                            id='editing-rows-button',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 5px 5px 0'}
                        ),
                    ], className="container"
                ),
            ], style={'flex': '1'}),
    ], style={'display': 'flex'})


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(content, name, date):
    if content is not None:
        children = [parse_contents(content, name, date)]
        return children


@app.callback(
    Output('data-table', 'columns'),
    [Input('editing-columns-button', 'n_clicks')],
    [State('editing-columns-name', 'value'),
     State('table-data', 'columns')])
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
    [State('table-data', 'data'),
     State('table-data', 'columns')])
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


@app.callback(
    Output('memory-std-xdata', 'data'),
    [Input('button-std-x-update', 'n_clicks')],
    [State('table-std-x', 'data')])
def update_std_x(n_clicks, xdata):
    if int(n_clicks) < 1:
        raise PreventUpdate

    for i, xi in enumerate(xdata):
        if isinstance(xi['x'], str):
            xdata[i]['x'] = xi['x'].replace(',', '.')

    new_xdata = sorted([float(item['x']) for item in xdata])

    return new_xdata


@app.callback(
    Output('table-std-x', 'data'),
    [Input('button-add-x-std', 'n_clicks')],
    [State('table-std-x', 'data'),
     State('table-std-x', 'columns')])
def add_row(n_clicks, rows, columns):
    if int(n_clicks) < 1:
        raise PreventUpdate

    rows.append(dict(x=0))
    return rows


@app.callback(
    [Output('dropdown-standard', 'value'),
     Output('dropdown-standard', 'options')],
    [Input('button-add-to-standard', 'n_clicks')],
    [State('table-data', 'selected_cells'),
     State('table-data', 'data'),
     State('dropdown-standard', 'value'),
     State('dropdown-standard', 'options')]
)
def update_standard_dropdown(n_clicks, selected_data, data, dropdown_values, dropdown_options):
    if int(n_clicks) < 1:
        raise PreventUpdate

    dropdown_values = [] if dropdown_values is None else dropdown_values
    dropdown_options = [] if dropdown_options is None else dropdown_options

    options = [option['value'] for option in dropdown_options]
    for selected_cell in selected_data:
        row = selected_cell['row']
        column_id = selected_cell['column_id']

        value = str(data[row][column_id])
        if value not in options:
            dropdown_options.append({'label': value, 'value': value})
        if value not in dropdown_values:
            dropdown_values.append(value)

    return dropdown_values, dropdown_options


@app.callback(
    Output('memory-std-ydata', 'data'),
    [Input('button-std-y-update', 'n_clicks')],
    [State('dropdown-standard', 'value')]
)
def update_std_y(n_clicks, ydata):
    if int(n_clicks) < 1:
        raise PreventUpdate

    return [float(yi) for yi in ydata]


@app.callback(
    Output('graph', 'figure'),
    [Input('memory-std-xdata', 'data'),
     Input('memory-std-ydata', 'data')]
)
def update_graph(xdata, ydata):
    if not xdata or not ydata:
        raise PreventUpdate
    if len(xdata) != len(ydata):
        raise PreventUpdate

    x_regression = np.arange(0, max(xdata) + 1, .1)
    model = FourParametricLogistic()
    model.fit(xdata, ydata)

    traces = [
        dict(
            x=xdata,
            y=ydata,
            mode='markers',
            opacity=0.7,
            marker={
                'size': 10,
            },
            name="Standard"
        ),
        dict(
            x=x_regression,
            y=model.predict(x_regression),
            mode='lines',
            name="Std curve"
        )
    ]

    layout = dict(
        annotations=[
            dict(
                x=.95,
                y=0.15,
                showarrow=False,
                text="R^2 = {}".format(np.round(model.r2(xdata, ydata), 4)),
                xref="paper",
                yref="paper"
            )
        ]
    )

    return dict(
        data=traces,
        layout=layout
    )


if __name__ == '__main__':
    app.run_server(debug=True)
