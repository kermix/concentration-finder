import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from regression import FourParametricLogistic, FourParametricLogisticEncoder
from tools import create_and_mix_color_scale

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='graph',
                      style=dict(height='calc(65vh - 8px)'),
                      figure=dict(
                          layout=dict(
                              yaxis=dict(
                                  scaleanchor="x",
                                  scaleratio=1,
                              )
                          )
                      )),
        ], style={'flex': '3'}),
        html.Div([
            html.Details(
                [
                    html.Summary("Concentrations of standard", style={'margin': '0 0 5px 0'}),
                    dcc.Dropdown(
                        placeholder="Concentrations of standard",
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
                    )
                ]
            ),
            html.Details(
                [
                    html.Summary("Absorbances of standard", style={'margin': '0 0 5px 0'}),
                    dcc.Dropdown(
                        placeholder="Standard curves",
                        id="dropdown-standards",
                        multi=True,
                        style={'margin': '0 0 5px 0'}
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                placeholder='New standard curve absorbances',
                                id="dropdown-new-standard",
                                multi=True,
                                style={'margin': '0 0 5px 0'}
                            ),
                        ], id="div-new-standard"
                    ),

                    dcc.Input(
                        id='input-new-standard',
                        placeholder='Enter new standard curve name',
                        autoComplete="off",
                        style={'margin': '5px 5px 0 0', 'float': 'left'}
                    ),
                    html.Button(
                        "Add",
                        id="button-std-y-add",
                        n_clicks=0,
                        style={'margin': "5px 0 0 0"}
                    ),
                ]
            ),
            html.Details(
                [
                    html.Summary('Standard and Model Details', style={'margin': '0 0 5px 0'}),
                    html.P(id='p-standard-model-details')
                ]
            ),
            html.Details(
                [
                    html.Summary("Data", style={'margin': '0 0 5px 0'}),
                    dcc.Dropdown(
                        placeholder="Data traces",
                        id="dropdown-traces",
                        multi=True,
                        style={'margin': '0 0 5px 0'}
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                placeholder='New data trace absorbances',
                                id="dropdown-new-trace",
                                multi=True,
                                style={'margin': '0 0 5px 0'})
                        ], id='div-new-trace'),
                    dcc.Input(
                        id='input-new-trace',
                        placeholder="Enter new data trace name",
                        autoComplete="off",
                        style={'margin': "5px 5px 0 0", 'float': 'left'}),
                    html.Button("Add",
                                id="button-trace-y-add",
                                n_clicks=0,
                                style={'margin': "5px 0 0 0"}),
                ],
            ),
            html.Details(
                [
                    html.Summary('Data Details', style={'margin': '0 0 5px 0'}),
                    html.P(id='p-data-details')
                ]
            ),
            html.Details(
                [
                    html.Summary('Export Analysis', style={'margin': '0 0 5px 0'}),
                ]
            ),
        ], style={'flex': '1', 'overflowY': 'auto'})
    ], style={'display': 'flex', 'height': 'calc(65vh - 8px)'}),
    html.Div(
        [
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': 'calc(100% - 10px)',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '5px'
                },
                multiple=False
            ),
        ],
        id='output-data-upload',
    ),
    dcc.Store(id='memory-std-xdata'),
    dcc.Store(id='memory-std-ydata'),
    dcc.Store(id='memory-standards'),
    dcc.Store(id='memory-models'),
    dcc.Store(id='memory-traces-ydata'),
    dcc.Store(id='memory-traces-xdata')
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
            ], style={'flex': '3', 'overflowY': 'auto'}
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
                            id='button-add-to-trace',
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
    ], style={'display': 'flex', 'height': 'calc(35vh - 8px)'})


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(content, name, date):
    if content is None:
        raise PreventUpdate

    children = [parse_contents(content, name, date)]
    return children


@app.callback(
    Output('table-data', 'columns'),
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
    Output('table-data', 'data'),
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
    [Output('dropdown-new-trace', 'value'),
     Output('dropdown-new-trace', 'options')],
    [Input('button-add-to-trace', 'n_clicks')],
    [State('table-data', 'selected_cells'),
     State('table-data', 'data'),
     State('dropdown-new-trace', 'value'),
     State('dropdown-new-trace', 'options')]
)
def update_trace_dropdown(n_clicks, selected_data, data, dropdown_values, dropdown_options):
    if int(n_clicks) < 1:
        raise PreventUpdate

    dropdown_values = [] if dropdown_values is None else dropdown_values
    dropdown_options = [] if dropdown_options is None else dropdown_options

    options = [option['value'] for option in dropdown_options]
    for selected_cell in selected_data:
        row = selected_cell['row']
        column_id = selected_cell['column_id']

        value = str(data[row][column_id])
        index = str(data[row]['Index']) + str(column_id)
        value = ';'.join((value, index))
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
def update_standards(n_clicks, new_standard_data, new_standard_name, current_standards, dropdown_values,
                     dropdown_options):
    if int(n_clicks) < 1 or not new_standard_name:
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
                   placeholder='New standard curve absorbances',
                   id="dropdown-new-standard",
                   multi=True,
                   style={'margin': '0 0 5px 0'},
                   value=[],
               )
           ], "", current_standards, dropdown_values, dropdown_options


@app.callback(
    [Output('div-new-trace', 'children'),
     Output('input-new-trace', 'value'),
     Output('memory-traces-ydata', 'data'),
     Output('dropdown-traces', 'value'),
     Output('dropdown-traces', 'options')],
    [Input('button-trace-y-add', 'n_clicks')],
    [State('dropdown-new-trace', 'value'),
     State('input-new-trace', 'value'),
     State('memory-traces-ydata', 'data'),
     State('dropdown-traces', 'value'),
     State('dropdown-traces', 'options')
     ]
)
def update_traces(n_clicks, new_trace_data, new_trace_name, current_traces, dropdown_values,
                  dropdown_options):
    if int(n_clicks) < 1 or not new_trace_name:
        raise PreventUpdate

    current_traces = {} if current_traces is None else current_traces

    if new_trace_name in current_traces.keys():
        raise PreventUpdate

    current_traces[new_trace_name] = [(float(yi.split(';')[0]), yi.split(';')[1]) for yi in new_trace_data]

    dropdown_values = [] if dropdown_values is None else dropdown_values
    dropdown_options = [] if dropdown_options is None else dropdown_options

    options = [option['value'] for option in dropdown_options]

    if new_trace_name not in dropdown_values:
        dropdown_values.append(new_trace_name)
    if new_trace_name not in options:
        dropdown_options.append({'label': new_trace_name, 'value': new_trace_name})

    return [
               dcc.Dropdown(
                   placeholder='New data trace absorbances',
                   id="dropdown-new-trace",
                   multi=True,
                   value=[],
                   style={'margin': '0 0 5px 0'})
           ], "", current_traces, dropdown_values, dropdown_options


@app.callback(
    [Output('graph', 'figure'),
     Output('memory-models', 'data')],
    [Input('dropdown-standards', 'value'),
     Input('memory-std-xdata', 'data'),
     Input('dropdown-traces', 'value')],
    [State('memory-standards', 'data'),
     State('memory-traces-ydata', 'data')]
)
def update_graph(choosen_standards, std_xdata, choosen_traces, std_ydata, traces_ydata):
    if not std_xdata or not std_ydata or not choosen_standards:
        raise PreventUpdate

    traces = []
    models = []
    x_regression = np.arange(0, max(std_xdata) + 1, .01)
    colors_scale = create_and_mix_color_scale(36)
    for std in choosen_standards:
        std_i = std_ydata[std]

        if len(std_i) != len(std_xdata):
            continue

        model = FourParametricLogistic()
        model.fit(std_xdata, std_i)

        r2_annotation = ["" for _ in range(len(x_regression))]
        r2_annotation[-1] = "R^2 = {}".format(np.round(model.r2(std_xdata, std_i), 4))

        models.append(FourParametricLogisticEncoder().encode(model))
        color = colors_scale.pop(0)
        traces.append(
            dict(
                x=std_xdata,
                y=std_i,
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'color': color
                },
                name=std,
                legendgroup=std,
            )
        )
        traces.append(
            dict(
                x=x_regression,
                y=model.predict(x_regression),
                text=r2_annotation,
                textposition="top left",
                mode='lines+text',
                line={
                    'color': color
                },
                name=f"{std} curve",
                legendgroup=std,
            )
        )

        if choosen_traces:
            for trace in choosen_traces:
                trace_color = colors_scale.pop(0)

                trace_i = traces_ydata[trace]
                trace_labels = [ti[1] for ti in trace_i]
                trace_values = [ti[0] for ti in trace_i]

                x = model.solve(trace_values)

                traces.append(
                    dict(
                        x=x,
                        y=trace_values,
                        text=trace_labels,
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 10,
                            'color': trace_color
                        },
                        name=f'{trace} vs {std}',
                    )
                )

    return dict(
        data=traces,
    ), models


if __name__ == '__main__':
    app.run_server(debug=True)
