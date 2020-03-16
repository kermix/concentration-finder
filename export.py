import pandas as pd
import json

from itertools import zip_longest

from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

center_alingment = Alignment(horizontal="center", vertical="center")
right_alingment = Alignment(horizontal="right", vertical="center")
bold_font = Font(name='Calibri', bold=True)
red_font = Font(name='Calibri', color='FFFF0000')


def jsonify_data_table(data):
    data_table = pd.DataFrame.from_dict(data, orient='columns')
    cols = list(data_table.columns.values)
    data_table = data_table[[cols[-1]] + cols[:-1]]

    return data_table.to_json()


def jsonify_models(standard_models, std_x, std_y):
    if standard_models is None:
        return ''

    models = dict()
    models["standard_concentrations"] = std_x
    models['data'] = {}
    for std_name in standard_models.keys():
        model = standard_models[std_name]
        A, B, C, D = model["A"], model["B"], model["C"], model["D"]
        R2 = model["R^2"]
        models['data'][f'{std_name}'] = \
            {
                'A': A, 'B': B, 'C': C, 'D': D, "R2": R2,
                'Y': std_y[std_name]
            }
    return json.dumps(models)


def jsonify_results(data):
    return json.dumps(data)


def write_models_data(ws, data):
    free_col = ws.max_column + 2
    free_row = 3

    header_cell = ws.cell(column=free_col, row=1, value="Concentrations of standard:")
    header_cell.alignment = right_alingment
    header_cell.font = Font(bold=True)
    ws.column_dimensions[header_cell.column_letter].width = str(len("Concentrations of standard"))
    for col, xi in enumerate(data['standard_concentrations'], free_col + 1):
        ws.cell(column=col, row=1, value=xi)

    for i, model_name in enumerate(data['data'].keys()):
        start_row = free_row + (8 * i)
        model = data['data'][model_name]

        model_name_cell = ws.cell(column=free_col, row=start_row, value=f'Model: {model_name}')
        ws.merge_cells(
            start_row=start_row,
            start_column=free_col,
            end_row=start_row,
            end_column=free_col+len(data['standard_concentrations'])
        )
        model_name_cell.font = bold_font
        model_name_cell.alignment = center_alingment

        y_name_cell = ws.cell(column=free_col, row=start_row + 1, value='Y:')
        y_name_cell.alignment=right_alingment
        y_name_cell.font=bold_font
        for j, yj in enumerate(model['Y'], 1):
            ws.cell(column=free_col+j, row=start_row+1, value=yj)

        parameters_name_cell = ws.cell(column=free_col, row=start_row + 4, value='Parameters:')
        parameters_name_cell.alignment=right_alingment
        parameters_name_cell.font=bold_font

        for j, value in enumerate(['A', 'B', 'C', 'D'], 1):
            parameter_name_cell = ws.cell(column=free_col + j, row=start_row+3, value=value)
            parameter_name_cell.alignment=right_alingment
            parameter_name_cell.font=bold_font

        for j, value in enumerate([model['A'], model['B'], model['C'], model['D']], 1):
            ws.cell(column=free_col + j, row=start_row +4, value=value)

        r2_name_cell = ws.cell(column=free_col, row=start_row + 6, value='R2:')
        r2_name_cell.alignment = right_alingment
        r2_name_cell.font = bold_font
        ws.cell(column=free_col + 1, row=start_row + 6, value=model['R2'])


def write_results_data(ws, data):
    traces_pairs = list(zip_longest(*[iter(data.keys())]*2, fillvalue=None))
    max_col = 1
    for pair in traces_pairs:
        start_row = max([cell.row for row in ws.iter_rows(min_col=max_col, max_col=max_col+6, values_only=False)
                         for cell in row if cell.value is not None], default=0)+2
        for i, trace in enumerate(pair):
            if trace is not None:
                x, y, labels = data[trace]['x'], data[trace]['y'], data[trace]['labels']
                start_col = max_col+(3*i)
                trace_name_cell = ws.cell(column=start_col, row=start_row, value=trace)
                ws.merge_cells(
                    start_row=start_row,
                    start_column=start_col,
                    end_row=start_row,
                    end_column=start_col + 2
                )
                trace_name_cell.alignment = center_alingment
                trace_name_cell.font = bold_font
                for j, col_label in enumerate(['Label:', 'X:', 'Y:']):
                    cell = ws.cell(column=start_col+j, row=start_row+1, value=col_label)
                    cell.alignment = center_alingment
                    cell.font = bold_font
                for j, xj, yj, labelj in zip(range(len(x)), x, y, labels):
                    ws.cell(column=start_col, row=start_row+2+j, value=labelj)
                    x_cell = ws.cell(column=start_col+1, row=start_row+2+j, value=xj)
                    if xj is None:
                        x_cell.value='Error'
                        x_cell.font=red_font
                    ws.cell(column=start_col+2, row=start_row+2+j, value=yj)


