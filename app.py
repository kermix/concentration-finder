import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.exceptions import PreventUpdate

import pandas as pd

from regression import FourParametricLogistic, FourParametricLogisticEncoder

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
                        dcc.Dropdown(
                            id='dropdown-concentrations',
                            multi=True,
                            style={'margin': '0 0 5px 0'}
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id='input-new-concentration',
                                    placeholder='Enter new concentration and press enter',
                                    debounce=True,
                                    autoComplete="off",
                                    style={'width': '100%', 'margin': '0 0 5px 0'}
                                ),
                            ],
                            id="div-new-concentration",
                        ),
                        html.Label(
                            [
                                "Set Standard",
                                dcc.Dropdown(
                                    id="dropdown-standards",
                                    multi=True,
                                    style={'margin': '0 0 5px 0'}
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="dropdown-new-standard",
                                            multi=True,
                                            style={'margin': '0 0 5px 0'}
                                        ),
                                    ], id="div-new-standard"
                                ),

                                dcc.Input(
                                    id='input-new-standard',
                                    placeholder='Enter new standard name',
                                    style={'width': '100%', 'margin': '0 0 5px 0'}
                                ),
                                html.Button(
                                    "Add standard",
                                    id="button-std-y-add",
                                    n_clicks=0,
                                    style={'margin': "5px 0 0 0"}
                                ),
                            ]
                        ),
                    ]
                )
            ], style={'flex': '1', 'padding': '0 10px'}),
            html.Div(
                [
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
                            html.Button("Add trace", style={'margin': "5px 0 0 0"}),
                        ],
                    ),
                ], style={'flex': '2', 'padding': '0 10px'})
        ], style={'flex': '1', 'display': 'flex', 'overflowY': 'auto'})
    ], style={'display': 'flex', 'height': '400px'}),
    html.Div(id='output-data-upload'),
    dcc.Store(id='memory-std-xdata'),
    dcc.Store(id='memory-std-ydata'),
    dcc.Store(id='memory-standards'),
    dcc.Store(id='memory-models'),

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
                        html.P("Manage Data", style={"text-align": "right"}),
                        html.Button(
                            'To Standard',
                            id='button-add-to-standard',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                        ),
                        html.Button(
                            'To data',
                            id='button-add-to trace',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                        )
                    ], className="container"
                ),
                html.Div(
                    [
                        html.P("Manage Table", style={"text-align": "right"}),
                        dcc.Input(
                            id='editing-columns-name',
                            placeholder='Enter a column name...',
                            value='',
                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                        ),
                        html.Button(
                            'Add Column',
                            id='editing-columns-button',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                        ),
                        html.Button(
                            'Add Row',
                            id='editing-rows-button',
                            n_clicks=0,
                            style={'float': 'right', 'margin': '0 0 5px 5px'}
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
    [Output('div-new-concentration', 'children'),
     Output('dropdown-concentrations', 'value'),
     Output('dropdown-concentrations', 'options')],
    [Input('input-new-concentration', 'value')],
    [State('dropdown-concentrations', 'value'),
     State('dropdown-concentrations', 'options')]
)
def add_concentration(concentation, dropdown_values, dropdown_options):
    dropdown_values = [] if dropdown_values is None else dropdown_values
    dropdown_options = [] if dropdown_options is None else dropdown_options

    try:
        concentation = float(concentation.replace(',', '.'))
    except (ValueError, AttributeError):
        raise PreventUpdate

    options = [option['value'] for option in dropdown_options]

    if concentation not in options:
        dropdown_options.append({'label': concentation, 'value': concentation})
    if concentation not in dropdown_values:
        dropdown_values.append(concentation)

    return [
               dcc.Input(
                   id='input-new-concentration',
                   placeholder='Enter new concentration and press enter',
                   debounce=True,
                   value="",
                   autoComplete='off',
                   style={'width': "100%"}
               )
           ], dropdown_values, dropdown_options


@app.callback(
    Output('memory-std-xdata', 'data'),
    [Input('dropdown-concentrations', 'value')])
def update_std_x(xdata):
    if xdata is None:
        raise PreventUpdate

    return sorted([float(item) for item in xdata])


@app.callback(
    [Output('dropdown-new-standard', 'value'),
     Output('dropdown-new-standard', 'options')],
    [Input('button-add-to-standard', 'n_clicks')],
    [State('table-data', 'selected_cells'),
     State('table-data', 'data'),
     State('dropdown-new-standard', 'value'),
     State('dropdown-new-standard', 'options')]
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
        dropdown_values.append(value)

    return dropdown_values, dropdown_options


@app.callback(
    [Output('div-new-standard', 'children'),
     Output('input-new-standard', 'value'),
     Output('memory-standards', 'data'),
     Output('dropdown-standards', 'value'),
     Output('dropdown-standards', 'options')],
    [Input('button-std-y-add', 'n_clicks')],
    [State('dropdown-new-standard', 'value'),
     State('input-new-standard', 'value'),
     State('memory-standards', 'data'),
     State('dropdown-standards', 'value'),
     State('dropdown-standards', 'options')
     ]
)
def update_standards(n_clicks, new_standard_data, new_standard_name, current_standards, dropdown_values, dropdown_options):
    if int(n_clicks) < 1:
        raise PreventUpdate

    current_standards = {} if current_standards is None else current_standards

    if new_standard_name in current_standards.keys():
        raise PreventUpdate

    current_standards[new_standard_name] = [float(yi) for yi in new_standard_data]

    dropdown_values = [] if dropdown_values is None else dropdown_values
    dropdown_options = [] if dropdown_options is None else dropdown_options

    options = [option['value'] for option in dropdown_options]

    if new_standard_name not in dropdown_values:
        dropdown_values.append(new_standard_name)
    if new_standard_name not in options:
        dropdown_options.append({'label': new_standard_name, 'value': new_standard_name})

    return [
               dcc.Dropdown(
                   id="dropdown-new-standard",
                   multi=True,
                   style={'margin': '0 0 5px 0'},
                   value=[],
               )
           ], "", current_standards, dropdown_values, dropdown_options


@app.callback(
    [Output('graph', 'figure'),
     Output('memory-models', 'data')],
    [Input('dropdown-standards', 'value'),
     Input('memory-std-xdata', 'data')],
    [State('memory-standards', 'data')]
)
def update_graph(choosen_standards, xdata, ydata):
    if not xdata or not ydata or not choosen_standards:
        raise PreventUpdate

    traces = []
    models = []
    x_regression = np.arange(0, max(xdata) + 1, .01)

    for std in choosen_standards:
        std_i = ydata[std]

        if len(std_i) != len(xdata):
            continue

        model = FourParametricLogistic()
        model.fit(xdata, std_i)


        r2_annotation = ["" for _ in range(len(x_regression))]
        r2_annotation[-1] = "R^2 = {}".format(np.round(model.r2(xdata, std_i), 4))

        models.append(FourParametricLogisticEncoder().encode(model))

        traces.append(
            dict(
                x=xdata,
                y=std_i,
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                },
                name=std,
                legendgroup=std
            )
        )
        traces.append(
            dict(
                x=x_regression,
                y=model.predict(x_regression),
                text=r2_annotation,
                textposition="top left",
                mode='lines+text',
                name=f"{std} curve",
                legendgroup=std
            )
        )

    return dict(
        data=traces,
    ), models


if __name__ == '__main__':
    app.run_server(debug=True)
