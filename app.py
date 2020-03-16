import base64
import io
import json
import os
import uuid
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import openpyxl
import pandas as pd
import redis
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
from flask import send_file

import export
from regression import FourParametricLogistic
from tools import create_and_mix_color_scale

r = redis.from_url(os.environ.get("REDIS_URL"))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='graph',
                      style=dict(height='60vh'),
                      figure=dict(
                          layout=dict(
                              yaxis=dict(
                                  scaleanchor="x",
                                  scaleratio=1,
                              )
                          )
                      )),
        ], style={'flex': '2'}),
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
                ], open=True
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
                                placeholder='1: New standard curve absorbances',
                                id="dropdown-new-standard-1",
                                multi=True,
                                style={'margin': '0 0 5px 0'}
                            ),
                            dcc.Dropdown(
                                placeholder='2: New standard curve absorbances',
                                id="dropdown-new-standard-2",
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
                ], id='standards'
            ),
            html.Details(
                [
                    html.Summary('Standard and Model Details', style={'margin': '0 0 5px 0'}),
                    html.Div(id='div-standard-model-details')
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
                                placeholder=f'{i} New data trace absorbances',
                                id=f"dropdown-new-trace-{i}",
                                multi=True,
                                style={'margin': '0 0 5px 0'}
                            ) for i in [1, 2]
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
                ], id='data'
            ),
            html.Details(
                [
                    html.Summary('Data Details', style={'margin': '0 0 5px 0'}),
                    html.Div(id='div-data-details')
                ]
            ),
            html.Details(
                [
                    html.Summary('Export Analysis', style={'margin': '0 0 5px 0'}),
                    html.Button("Generate", id='gen-export', n_clicks=0),
                    html.A("Download", id='download-export', href='#', target='_blank')

                ]
            ),
        ], style={'flex': '1', 'overflowY': 'auto', 'margin': '5px 0'})
    ], style={'display': 'flex', 'height': '60vh'}),
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
                multiple=True
            ),
            dash_table.DataTable(id='table-data')
        ],
        id='output-data-upload',
    ),
    dcc.Store(id='memory-std-xdata'),
    dcc.Store(id='memory-std-ydata'),  # MOVE choosen standadards in to Store
    dcc.Store(id='memory-standards'),
    dcc.Store(id='memory-models'),
    dcc.Store(id='memory-traces-ydata'),
    dcc.Store(id='memory-traces-xdata'),
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
            df = pd.read_excel(io.BytesIO(decoded), skiprows=5, index_col=0)
            df = remove_empty(df)
            # df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)4
            return df
    except Exception as e:
        print(e)
    raise PreventUpdate


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise PreventUpdate

    dfs = [
        parse_contents(c, n, d) for c, n, d in
        zip(list_of_contents, list_of_names, list_of_dates)
    ]
    df_concat = pd.concat(dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean().round(4).reset_index()
    return html.Div([
        html.Div(
            [
                dash_table.DataTable(
                    data=df_means.to_dict('records'),
                    columns=[{"name": str(i), "id": str(i)} for i in df_means.columns],
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
                        html.Details(
                            [
                                html.Summary("Add to Standard", style={"text-align": "right"}),
                                html.Div(
                                    [
                                        html.Button(
                                            'To Standard 1',
                                            id='button-add-to-standard-1',
                                            n_clicks=0,
                                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                                        ),
                                        html.Button(
                                            'To Standard 2',
                                            id='button-add-to-standard-2',
                                            n_clicks=0,
                                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                                        ),
                                    ], className='container w-100'
                                )
                            ], open=True
                        ),
                        html.Details(
                            [
                                html.Summary('Add to data', style={"text-align": "right"}),
                                html.Div(
                                    [
                                        html.Button(
                                            'To data 1',
                                            id='button-add-to-trace-1',
                                            n_clicks=0,
                                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                                        ),
                                        html.Button(
                                            'To data 2',
                                            id='button-add-to-trace-2',
                                            n_clicks=0,
                                            style={'float': 'right', 'margin': '0 0 5px 5px'}
                                        )
                                    ], className='container w-100'
                                )
                            ], open=True
                        )

                    ], className="container"
                ),
                html.Div(
                    [
                        html.Details([
                            html.Summary("Manage Table", style={"text-align": "right"}),
                            html.Div(
                                [

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
                                    )
                                ], className='container w-100'
                            )
                        ])
                    ], className="container"
                ),
            ], style={'flex': '1'}),
    ], style={'display': 'flex', 'height': '40vh'})


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

    return [float(item) for item in xdata]


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


for i in [1, 2]:
    app.callback(
        [Output(f'dropdown-new-standard-{i}', 'value'),
         Output(f'dropdown-new-standard-{i}', 'options')],
        [Input(f'button-add-to-standard-{i}', 'n_clicks')],
        [State('table-data', 'selected_cells'),
         State('table-data', 'data'),
         State(f'dropdown-new-standard-{i}', 'value'),
         State(f'dropdown-new-standard-{i}', 'options')]
    )(update_standard_dropdown)

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
        index = str(data[row]['index']) + str(column_id)
        value = ';'.join((value, index))
        if value not in options:
            dropdown_options.append({'label': value, 'value': value})
        dropdown_values.append(value)

    return dropdown_values, dropdown_options


for i in [1, 2]:
    app.callback(
        [Output(f'dropdown-new-trace-{i}', 'value'),
         Output(f'dropdown-new-trace-{i}', 'options')],
        [Input(f'button-add-to-trace-{i}', 'n_clicks')],
        [State('table-data', 'selected_cells'),
         State('table-data', 'data'),
         State(f'dropdown-new-trace-{i}', 'value'),
         State(f'dropdown-new-trace-{i}', 'options')]
    )(update_trace_dropdown)


@app.callback(
    [Output('div-new-standard', 'children'),
     Output('input-new-standard', 'value'),
     Output('memory-standards', 'data'),
     Output('dropdown-standards', 'value'),
     Output('dropdown-standards', 'options')],
    [Input('button-std-y-add', 'n_clicks')],
    [State('dropdown-new-standard-1', 'value'),
     State('dropdown-new-standard-2', 'value'),
     State('input-new-standard', 'value'),
     State('memory-standards', 'data'),
     State('dropdown-standards', 'value'),
     State('dropdown-standards', 'options'),
     ]
)
def update_standards(n_clicks, new_standard_data_1, new_standard_data_2, new_standard_name, current_standards,
                     dropdown_values, dropdown_options):
    if int(n_clicks) < 1 or not new_standard_name:
        raise PreventUpdate

    if len(new_standard_data_1) != len(new_standard_data_2):
        raise PreventUpdate

    current_standards = {} if current_standards is None else current_standards

    if new_standard_name in current_standards.keys():
        raise PreventUpdate

    current_standards[new_standard_name] = [(yi + yj) / 2.0 for yi, yj in zip(
        [float(yi) for yi in new_standard_data_1],
        [float(yi) for yi in new_standard_data_2]
    )]

    dropdown_values = [] if dropdown_values is None else dropdown_values
    dropdown_options = [] if dropdown_options is None else dropdown_options

    options = [option['value'] for option in dropdown_options]

    if new_standard_name not in dropdown_values:
        dropdown_values.append(new_standard_name)
    if new_standard_name not in options:
        dropdown_options.append({'label': new_standard_name, 'value': new_standard_name})

    return [
               dcc.Dropdown(
                   placeholder=f'{i}: New standard curve absorbances',
                   id=f"dropdown-new-standard-{i}",
                   multi=True,
                   style={'margin': '0 0 5px 0'},
                   value=[],
               ) for i in [1, 2]
           ], "", current_standards, dropdown_values, dropdown_options


@app.callback(
    [Output('div-new-trace', 'children'),
     Output('input-new-trace', 'value'),
     Output('memory-traces-ydata', 'data'),
     Output('dropdown-traces', 'value'),
     Output('dropdown-traces', 'options')],
    [Input('button-trace-y-add', 'n_clicks')],
    [State('dropdown-new-trace-1', 'value'),
     State('dropdown-new-trace-2', 'value'),
     State('input-new-trace', 'value'),
     State('memory-traces-ydata', 'data'),
     State('dropdown-traces', 'value'),
     State('dropdown-traces', 'options')
     ]
)
def update_traces(n_clicks, new_trace_data_1, new_trace_data_2, new_trace_name, current_traces, dropdown_values,
                  dropdown_options):
    if int(n_clicks) < 1 or not new_trace_name:
        raise PreventUpdate

    current_traces = {} if current_traces is None else current_traces

    if new_trace_name in current_traces.keys():
        raise PreventUpdate

    new_data_1 = [(float(yi.split(';')[0]), yi.split(';')[1]) for yi in new_trace_data_1]
    new_data_2 = [(float(yi.split(';')[0]), yi.split(';')[1]) for yi in new_trace_data_2]

    current_traces[new_trace_name] = [((yi[0] + yj[0]) / 2.0, ''.join((yi[1], yj[1]))) for yi, yj in
                                      zip(new_data_1, new_data_2)]

    dropdown_values = [] if dropdown_values is None else dropdown_values
    dropdown_options = [] if dropdown_options is None else dropdown_options

    options = [option['value'] for option in dropdown_options]

    if new_trace_name not in dropdown_values:
        dropdown_values.append(new_trace_name)
    if new_trace_name not in options:
        dropdown_options.append({'label': new_trace_name, 'value': new_trace_name})

    return [
               dcc.Dropdown(
                   placeholder=f'{i}: New data trace absorbances',
                   id=f"dropdown-new-trace-{i}",
                   multi=True,
                   value=[],
                   style={'margin': '0 0 5px 0'}
               ) for i in [1, 2]
           ], "", current_traces, dropdown_values, dropdown_options


@app.callback(
    [Output('graph', 'figure'),
     Output('memory-models', 'data'),
     Output('memory-traces-xdata', 'data')],
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
    models = {}
    traces_xdata = defaultdict(dict)
    x_regression = np.arange(0, max(std_xdata) + 1, .01)
    colors_scale = create_and_mix_color_scale(36)
    for std in choosen_standards:
        std_i = std_ydata[std]

        if len(std_i) != len(std_xdata):
            continue

        model = FourParametricLogistic()
        model.fit(std_xdata, std_i)

        r2 = model.r2(std_xdata, std_i)
        r2_annotation = ["" for _ in range(len(x_regression))]
        r2_annotation[-1] = "R^2 = {}".format(np.round(r2, 4))

        A, B, C, D = model.parameters
        models[std] = {"A": A, "B": B, "C": C, "D": D, "R^2": r2}
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
                color_index = len(choosen_standards) if len(choosen_standards) <= len(colors_scale) else -1
                trace_color = colors_scale.pop(color_index)

                trace_i = traces_ydata[trace]
                trace_labels = [ti[1] for ti in trace_i]
                trace_values = [ti[0] for ti in trace_i]

                x = model.solve(trace_values)
                traces_xdata[f'{trace} vs {std}']['x'] = x
                traces_xdata[f'{trace} vs {std}']['y'] = trace_values
                traces_xdata[f'{trace} vs {std}']['labels'] = trace_labels


                traces.append(
                    dict(
                        x=x,
                        y=trace_values,
                        text=trace_labels,
                        mode='markers',
                        opacity=0.95,
                        marker={
                            'size': 10,
                            'color': trace_color
                        },
                        name=f'{trace} vs {std}',
                    )
                )

    return dict(
        data=traces,
    ), models, traces_xdata


@app.callback(
    [Output('div-standard-model-details', 'children'),
     Output('div-data-details', 'children')],
    [Input('memory-models', 'data'),
     Input('memory-std-xdata', 'data'),
     Input('memory-traces-xdata', 'data')],
    [State('memory-standards', 'data')]
)
def get_details(models, std_x, traces_data, std_y):
    content_std_models, content_data = [], []

    if std_x:
        content_std_models.append(
            html.Div(
                [
                    html.B('X: '),
                    f'{std_x}'
                ]
            )
        )

    if models:
        for key in models.keys():
            model = models[key]
            A, B, C, D = model["A"], model["B"], model["C"], model["D"]
            R2 = model["R^2"]
            content_std_models.append(
                html.Div(
                    [
                        html.Div([html.B(f'{key}:')]),
                        html.Div(
                            [
                                html.B('Y:'), f'{std_y[key]}', html.Br(),
                                html.B('Params: '), f'A={A:.3f}; B={B:.3f}; C={C:.3f}; D={R2:.3f}', html.Br(),
                                html.B(f'Stats: '), f'R^2={R2:.3f}'
                            ], style={'margin': '0 0 0 10px'}
                        )
                    ]
                )
            )

    if traces_data:
        for key in traces_data.keys():
            trace = traces_data[key]
            for i, xi in enumerate(trace['x']):
                if isinstance(xi, float):
                    trace['x'][i] = round(xi, 3)
            content_data.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.B(f'{key}:')
                            ]
                        ),
                        html.Div(
                            [
                                html.B('X: '), f'{trace["x"]}', html.Br(),
                                html.B('Y: '), f'{trace["y"]}'
                            ], style={'margin': '0 0 0 10px'}
                        )
                    ]
                )
            )

    return content_std_models, content_data


for output, button_1, button_2 in [
    ['standards', 'button-add-to-standard-1', 'button-add-to-standard-2'],
    ['data', 'button-add-to-trace-1', 'button-add-to-trace-2']
]:
    app.clientside_callback(
        output=Output(output, 'open'),
        inputs=[Input(button_1, 'n_clicks'),
                Input(button_2, 'n_clicks')],
        clientside_function = ClientsideFunction(
            namespace='clientside',
            function_name ='open_details_on_btn_click'
        )
    )

@app.callback(
    Output('download-export', 'href'),
    [Input('gen-export', 'n_clicks')],
    [State('table-data', 'data'),
     State('memory-models', 'data'),
     State('memory-std-xdata', 'data'),
     State('memory-standards', 'data'),
     State('memory-traces-xdata', 'data')]
)
def generate_export(n_clicks, data_table_data, models, std_x, std_y, traces_data):
    if n_clicks > 0:
        export_data = dict()

        export_data["data_table"] = export.jsonify_data_table(data_table_data)
        export_data["models"] = export.jsonify_models(models, std_x, std_y)
        export_data['results'] = export.jsonify_results(traces_data)

        data_id = str(uuid.uuid4())
        r.append(data_id, json.dumps(export_data))
        return f"download/{data_id}"
    return '#'


@app.server.route('/download/<path:path>')
def download_export(path):
    export_data = json.loads(r.get(path))

    file_io = io.BytesIO()
    wb = openpyxl.Workbook()
    writer = pd.ExcelWriter(file_io, engine='openpyxl')
    writer.book = wb
    writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)

    df = pd.read_json(export_data['data_table']).set_index('index')
    df.to_excel(writer, sheet_name='Data')
    writer.save()

    wb.active = wb['Data']
    ws = wb.active

    export_results_data = json.loads(export_data['results'])
    export.write_results_data(ws, export_results_data)
    writer.save()

    export_model_data = json.loads(export_data["models"])
    export.write_models_data(ws, export_model_data)
    writer.save()

    wb.remove(wb['Sheet'])
    writer.save()
    file_io.seek(0)
    return send_file(file_io,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     attachment_filename=f'{path}.xlsx',
                     as_attachment=True)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
