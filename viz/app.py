import os
import copy
import time
import datetime

import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from cdr.viz.layout_helper import run_standalone_app

from cdr.util import load_cdr

# UPLOAD_DIRECTORY = '/cdr/uploaded_files'
# if not os.path.exists(UPLOAD_DIRECTORY):
#    os.makedirs(UPLOD_DIRECTORY)

def model_generation(model_path):
    return load_cdr(model_path)

def description():
    return 'Visualization of continuous-time deconvolutional regression (CDR) neural networks: a regression technique for temporarily diffuse effects.'

def header_colors():
    return {
        'bg_color': '#015BB0',
        'font_color': '#FFFFFF',
        'light_logo': True
    }

def generate_figure(x_value, y_value, response_value, resparam_value, ci, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
    output_tuple = model.get_plot_data(
        ref_varies_with_x=False, 
        xvar=x_value, 
        yvar=y_value,
        # responses =
        # response_params = 
        pair_manipulations=True,
        level=ci,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        n_samples=10)
    # print('response value below')
    # print(response_value)
    # print('response param below')
    # print(resparam_value)
    # print('first key below')
    # first_key = list(resparam_value)[0]
    # print(first_key)
    # print('keys below')
    # print(list(resparam_value))
    # print('first value below')
    # first_value = resparam_value[first_key]
    # print(first_value)
    # print('---')
    # test = output_tuple[1]
    # print(test)
    # print(test[response_value]['logit.N'])
    # print(test[response_value])
    zmin = zmin
    zmax = zmax
    x, y = output_tuple[0]
    d = output_tuple[1]
    print(d)
    z = d[response_value][resparam_value]
    z1 = z[...,0]   
    min_d = output_tuple[2]
    min_z = min_d[response_value][resparam_value]
    z2 = min_z[...,0]
    max_d = output_tuple[3]
    max_z = max_d[response_value][resparam_value]
    z3 = max_z[...,0]

    fig = go.Figure(data=[
        # go.Surface(z=z1, x=x, y=y, colorscale='blues'),
        go.Surface(z=z1, x=x, y=y, colorscale=[
            [0.0, 'rgb(255, 0, 0)'],
            [0.5, 'rgb(192, 192, 192)'],
            [1.0, 'rgb(0, 0, 204)']
        ]),
        go.Surface(z=z2, x=x, y=y, colorscale='greys', showscale=False, opacity=0.5),
        go.Surface(z=z3, x=x, y=y, colorscale='greys', showscale=False, opacity=0.5)
    ])
    
    fig.update_layout(
        font_family='Times',
        title_font_family='Times',
        title='Surface Plot of ' + y_value + ' vs. ' + x_value,
        scene = dict(
            xaxis_title=x_value,
            yaxis_title=y_value,
            zaxis_title=response_value + " " + resparam_value,
            xaxis=dict(range=[xmin, xmax]),
            yaxis=dict(range=[ymin, ymax]),
            zaxis=dict(range=[zmin, zmax])
        ),
        autosize=False,
        width=650, height=650,
        margin=dict(l=100, r=100, b=100, t=100)
    )
    return fig

def generate_2d(x_value, response_value, resparam_value, xmin=None, xmax=None):
    output_tuple_2d = model.get_plot_data(
        ref_varies_with_x=True,
        xvar=x_value,
        xmin=xmin,
        xmax=xmax,
        manipulations=manipulations)
    x2d = output_tuple_2d[0]
    d2d = output_tuple_2d[1]
    y2d = d2d[response_value][resparam_value]
    y2d_splice = y2d[...,0]
    # y1 = y[...,0]
    # y2 = y[...,1]
    fig = go.Figure(data=[
        go.Scatter(x=x2d, y=y2d_splice, marker=dict(color='blue'), mode='lines'),
        go.Scatter(x=x2d, y=y2d_splice, marker=dict(color='blue'))
    ])

    if xmin is not None and xmax is not None:
        fig.update_xaxes(range=[xmin, xmax])
    # fig.add_trace(go.Scatter(x=x, y=y1, mode='lines'))
    # fig.add_trace(go.Scatter(x=x, y=y2, mode='lines'))
    fig.update_layout(
        font_family='Times',
        title_font_family='Times',
        title='2D',
        xaxis_title=x_value,
        yaxis_title=response_value + " " + resparam_value,
        autosize=False,
        width=650, height=650,
        margin=dict(l=100, r=100, b=100, t=100)
    )
    return fig

def layout():
    return html.Div(
        id='cdrnn-body',
        className='app-body',
        children=[
            html.Div(
                id='cdrnn-control-tabs',
                className='control-tabs',
                children=[
                    dcc.Tabs(id='cdrnn-tabs', value='what-is', children=[
                        dcc.Tab(
                            label='About',
                            value='what-is',
                            children=html.Div(className='control-tab', children=[
                                html.H4(className='what-is', children='What is CDR(NN) Viewer?'),
                                html.P('CDR(NN) Viewer is a visualization tool for continuous-time deconvolutional regression (CDR) neural networks. You can get a high-level overview here: https://github.com/coryshain/cdr/blob/master/README.md and full documentation here: https://cdr.readthedocs.io/en/latest/.'),
                                html.P('In the "View 3D/2D" tabs, you can use the dropdown menus to change plotted variables such as responses, multivariate response parameters, signals, and more. You can also use the text inputs to change the range of various axes.')
                            ])
                        ),
                        dcc.Tab(
                            label='Model',
                            value='model-type',
                            children=html.Div(className='control-tab', children=[
                                html.Div(
                                    title='Select CDRNN model to view',
                                    className='app-controls-block',
                                    children=[
                                        html.Div(
                                            className='fullwidth-app-controls-name',
                                            children='Model type'),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='The CDR(NN) model being viewed is: ' + args.model
                                        ),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Currently, in order to change the CDR(NN) model you have to restart the command terminal and input the model path from there.'
                                        )
                                    ]
                                )
                            ])
                        ),
                        dcc.Tab(
                            label='View 3D',
                            children=html.Div(className='control-tab', children=[
                                html.Div(
                                    title='Change respective x-y coordinate axes to view',
                                    className='app-controls-block',
                                    children=[
                                        html.Div(
                                            className='fullwidth-app-controls-name',
                                            children='Change respective axes names to view'),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Select the respective x-y coordinate axes names ' +
                                            'by using the dropdown menus below.'
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_x',
                                            options=[ 
                                                {'label': i, 'value': i} for i in options],
                                            value=options[0]
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_y',
                                            options=[
                                                {'label': i, 'value': i} for i in options],
                                            value=options[len(options) - 1]
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_response',
                                            options=[
                                                {'label': i, 'value': i} for i in response_options],
                                            value=response_options[0]
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_resparams',
                                            options=[
                                                {'label': i, 'value': i} for i in model.get_response_params(response_options[0])],
                                            value=model.get_response_params(response_options[0])[0]
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_resparams_multivariate',
                                            options=[
                                                {'label': i, 'value': i} for i in model._expand_param_name_by_dim(response_options[0], model.get_response_params(response_options[0]))],
                                            value=model._expand_param_name_by_dim(response_options[0], model.get_response_params(response_options[0])[0])[0]
                                        )
                                    ]
                                ),
                                html.Div(
                                    title='Change the bounds of x-y coordinate axes',
                                    className='app-controls-block',
                                    children=[
                                        html.Div(className='app-controls-name',
                                                 children='Bounds manipulator'),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Minimum and maximum of x-coordinate axis'
                                        ),
                                        dcc.Input(
                                            id='x_min',
                                            type='number',
                                            debounce=True,
                                            placeholder='Minimum x-axis value'
                                        ),
                                        dcc.Input(
                                            id='x_max',
                                            type='number',
                                            debounce=True,
                                            placeholder='Maximum x-axis value'
                                        ),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Minimum and maximum of y-coordinate axis'
                                        ),
                                        dcc.Input(
                                            id='y_min',
                                            type='number',
                                            debounce=True,
                                            placeholder='Minimum y-axis value'
                                        ),
                                        dcc.Input(
                                            id='y_max',
                                            type='number',
                                            debounce=True,
                                            placeholder='Maximum y-axis value'
                                        ),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Minimum and maximum of z-coordinate axis'
                                        ),
                                        dcc.Input(
                                            id='z_min',
                                            type='number',
                                            debounce=True,
                                            placeholder='Minimum z-axis value'
                                        ),
                                        dcc.Input(
                                            id='z_max',
                                            type='number',
                                            debounce=True,
                                            placeholder='Maximum z-axis value'
                                        )
                                    ]
                                ),
                                html.Div(
                                    title='Change the error interval',
                                    className='app-controls-block',
                                    children=[
                                        html.Div(className='app-controls-name',
                                                 children='Error interval'),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Adjust error interval from 0-100 (default: 95)'
                                        ),
                                        dcc.Input(
                                            id='ci',
                                            type='number',
                                            debounce=True,
                                            placeholder='Error interval',
                                            min=0, 
                                            max=100,
                                            step=1,
                                            value=95
                                        )
                                    ]
                                )
                            ])
                        ),
                        dcc.Tab(
                            label='View 2D',
                            children=html.Div(className='control-tab', children=[
                                html.Div(
                                    title='Change x coordinate axis to view',
                                    className='app-controls-block',
                                    children=[
                                        html.Div(className='fullwidth-app-controls-name',
                                                 children='Change x coordinate axis to view'),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Select the respective x coordinate axis name by ' +
                                            'using the dropdown menu below.'
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_x_2d',
                                            options=[ 
                                                {'label': i, 'value': i} for i in options],
                                            value=options[0]
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_response_2d',
                                            options=[
                                                {'label': i, 'value': i} for i in response_options],
                                            value=response_options[0]
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_resparams_2d',
                                            options=[
                                                {'label': i, 'value': i} for i in model.get_response_params(response_options[0])],
                                            value=model.get_response_params(response_options[0])[0]
                                        ),
                                        dcc.Dropdown(
                                            id='dropdown_resparams_multivariate_2d',
                                            options=[
                                                {'label': i, 'value': i} for i in model._expand_param_name_by_dim(response_options[0], model.get_response_params(response_options[0]))],
                                            value=model._expand_param_name_by_dim(response_options[0], model.get_response_params(response_options[0])[0])[0]
                                        )
                                    ]
                                ),
                                html.Div(
                                    title='Change the bounds of the x coordinate axis',
                                    className='app-controls-block',
                                    children=[
                                        html.Div(className='app-controls-name',
                                                 children='X-axis range'),
                                        html.Div(
                                            className='app-controls-desc',
                                            children='Minimum and maximum of x-coordinate axis'
                                        ),
                                        dcc.Input(
                                            id='x_min_2d',
                                            type='number',
                                            debounce=True,
                                            placeholder='Minimum x-axis value'
                                        ),
                                        dcc.Input(
                                            id='x_max_2d',
                                            type='number',
                                            debounce=True,
                                            placeholder='Maximum x-axis value'
                                        )
                                    ]
                                )
                            ])
                        )
                    ])
                ], style={'display': 'inline-block'}
            ), 
            html.Div(id='cdrnn3d-container', children=[
                dcc.Graph(id='graph-change', figure=generate_figure(options[0], options[len(options) - 1], response_options[0], model._expand_param_name_by_dim(response_options[0], model.get_response_params(response_options[0])[0])[0], 95))
            ], style={'display': 'inline-block', 'padding': 50}
            ),
            html.Div(id='cdrnn2d-continer', children=[
                dcc.Graph(id='2d', figure=generate_2d(options[0], response_options[0], model._expand_param_name_by_dim(response_options[0], model.get_response_params(response_options[0])[0])[0]))
            ], style={'display': 'inline-block', 'padding': 50})
        ]
    )

def callbacks(_app):
    @_app.callback( 
        Output('graph-change', 'figure'), 
        Input('dropdown_x', 'value'), 
        Input('dropdown_y', 'value'),
        Input('dropdown_response', 'value'),
        Input('dropdown_resparams_multivariate', 'value'),
        Input('ci', 'value'),
        Input('x_min', 'value'),
        Input('x_max', 'value'),
        Input('y_min', 'value'),
        Input('y_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value'),
    )
    def update_graph(x_value, y_value, response_value, resparam_value, level_value, x_min_value, x_max_value, y_min_value, y_max_value, z_min_value, z_max_value):
        return generate_figure(x_value, y_value, response_value, resparam_value, level_value, x_min_value, x_max_value, y_min_value, y_max_value, z_min_value, z_max_value)

    @_app.callback(
        Output('2d', 'figure'),
        Input('dropdown_x_2d', 'value'),
        Input('dropdown_response_2d', 'value'),
        Input('dropdown_resparams_multivariate_2d', 'value'),
        Input('x_min_2d', 'value'),
        Input('x_max_2d', 'value')
    )
    def update_2d(x_value, response_value, resparam_value, x_min_2d_value, x_max_2d_value):
        return generate_2d(x_value, response_value, resparam_value, x_min_2d_value, x_max_2d_value)

    @_app.callback(
        Output('dropdown_resparams', 'options'),
        Input('dropdown_response', 'value')
    )
    def update_response_param_options(response_value):
        new_response_params = [{'label': i, 'value': i} for i in model.get_response_params(response_value)]
        return new_response_params

    @_app.callback(
        Output('dropdown_resparams', 'value'),
        Input('dropdown_response', 'value')
    )
    def update_response_param_value(response_value):
        return model.get_response_params(response_value)[0]

    @_app.callback(
        Output('dropdown_resparams_multivariate', 'options'),
        Input('dropdown_response', 'value'),
        Input('dropdown_resparams', 'value'),
    )
    def update_multivariate(response_value, resparam_value):
        new_multivariate = [{'label': i, 'value': i} for i in model._expand_param_name_by_dim(response_value, resparam_value)]
        return new_multivariate

    @_app.callback(
        Output('dropdown_resparams_multivariate', 'value'),
        Input('dropdown_response', 'value'),
        Input('dropdown_resparams', 'value'),
    )
    def update_multivariate(response_value, resparam_value):
        return model._expand_param_name_by_dim(response_value, resparam_value)[0]

    @_app.callback(
        Output('dropdown_resparams_2d', 'options'),
        Input('dropdown_response_2d', 'value')
    )
    def update_response_param_options(response_value):
        new_response_params = [{'label': i, 'value': i} for i in model.get_response_params(response_value)]
        return new_response_params

    @_app.callback(
        Output('dropdown_resparams_2d', 'value'),
        Input('dropdown_response_2d', 'value')
    )
    def update_response_param_value(response_value):
        return model.get_response_params(response_value)[0]

    @_app.callback(
        Output('dropdown_resparams_multivariate_2d', 'options'),
        Input('dropdown_response_2d', 'value'),
        Input('dropdown_resparams_2d', 'value'),
    )
    def update_multivariate(response_value, resparam_value):
        new_multivariate = [{'label': i, 'value': i} for i in model._expand_param_name_by_dim(response_value, resparam_value)]
        return new_multivariate

    @_app.callback(
        Output('dropdown_resparams_multivariate_2d', 'value'),
        Input('dropdown_response_2d', 'value'),
        Input('dropdown_resparams_2d', 'value'),
    )
    def update_multivariate(response_value, resparam_value):
        return model._expand_param_name_by_dim(response_value, resparam_value)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    model = model_generation(args.model)
    model.set_predict_mode(True)
    options = model.impulse_names + ['t_delta', 'X_time']
    response_options = model.response_names
    print(response_options)
    print(model.get_response_params(response_options[0]))
    print(model.get_response_params(response_options[0])[0])
    print('----')
    response = response_options[0]
    response_param = model.get_response_params(response)[0]
    response_param = model._expand_param_name_by_dim(response, response_param)
    print(response_param)
    manipulations = [{model.impulse_names[0]: 1}]
    app = run_standalone_app(layout, callbacks, header_colors, __file__)
    server = app.server
    app.run_server(debug=True, port=5000)