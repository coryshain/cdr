import os
import copy
import time
import datetime

import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from cdr.viz.layout_helper import run_standalone_app

from cdr.util import load_cdr, get_irf_name

# UPLOAD_DIRECTORY = '/cdr/uploaded_files'
# if not os.path.exists(UPLOAD_DIRECTORY):
#    os.makedirs(UPLOD_DIRECTORY)

def model_generation(model_path):
    return load_cdr(model_path)

def description():
    return 'Visualization of continuous-time deconvolutional regression (CDR) neural networks: a regression technique for temporarily diffuse effects.'

def get_resparams(model, response):
    resparams = []
    if response in model.response_names:
        for x in model.get_response_params(response):
            for y in model.expand_param_name(response, x):
                resparams.append(y)
    return resparams

def generate_figure(
        xvar,
        response,
        resparam,
        yvar=None,
        n_samples=10,
        level=95,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
        X_ref=None
):
    plot_data = model.get_plot_data(
        ref_varies_with_x=xvar in ('t_delta', 'X_time') and yvar is not None,
        ref_varies_with_y=yvar in ('t_delta', 'X_time'),
        xvar=xvar,
        yvar=yvar,
        responses=response,
        response_params=resparam,
        X_ref=X_ref,
        pair_manipulations=True,
        level=level,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        n_samples=n_samples
    )

    if yvar is None: # 2D plot
        x2d = plot_data[0]
        d2d = plot_data[1]
        y2d = d2d[response][resparam]
        y2d_splice = y2d[..., 0]
        y_lower = plot_data[2][response][resparam][..., 0]
        y_upper = plot_data[3][response][resparam][..., 0]
        # y1 = y[...,0]
        # y2 = y[...,1]
        fig = go.Figure(data=[
            go.Scatter(x=x2d, y=y2d_splice, marker=dict(color='blue'), mode='lines'),
            go.Scatter(
                name='Upper Bound',
                x=x2d,
                y=y_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=x2d,
                y=y_lower,
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(0, 0, 255, 0.2)',
                fill='tonexty',
                showlegend=False
            )
        ])

        if xmin is not None and xmax is not None:
            fig.update_xaxes(range=[xmin, xmax])
        fig.update_layout(
            font_family='Helvetica',
            title_font_family='Helvetica',
            title='2D',
            xaxis_title=xvar,
            yaxis_title=response + " " + resparam,
            xaxis=dict(range=[xmin, xmax], gridcolor='rgb(200, 200, 200)'),
            yaxis=dict(gridcolor='rgb(200, 200, 200)'),
            plot_bgcolor='rgb(255, 255, 255)',
            paper_bgcolor='rgb(255, 255, 255)'
        )
    else: # 3D plot
        zmin = zmin
        zmax = zmax
        x, y = plot_data[0]
        d = plot_data[1]
        z = d[response][resparam]
        z1 = z[...,0]
        min_d = plot_data[2]
        min_z = min_d[response][resparam]
        z2 = min_z[...,0]
        max_d = plot_data[3]
        max_z = max_d[response][resparam]
        z3 = max_z[...,0]

        blue = np.array((0, 0, 255))
        red = np.array((255, 0, 0))
        gray = np.array((220, 220, 220))

        lower = z1.min()
        upper = z1.max()
        mag = max(np.abs(upper), np.abs(lower))
        lower_p = lower / mag
        upper_p = upper / mag
        if lower_p < 0:
            lower_c = blue * (-lower_p) + gray * (1 + lower_p)
        else:
            lower_c = red * lower_p + gray * (1 - lower_p)
        if upper_p > 0:
            upper_c = red * upper_p + gray * (1 - upper_p)
        else:
            upper_c = blue * (-upper_p) + gray * (1 + upper_p)

        colorscale = [
            (0., 'rgb(%s, %s, %s)' % tuple(lower_c)),
            (1., 'rgb(%s, %s, %s)' % tuple(upper_c)),
        ]

        if lower_p < 0 and upper_p > 0:
            midpoint = (-lower_p) / (upper_p - lower_p)
            colorscale.insert(1, (midpoint, 'rgb(%s, %s, %s)' % tuple(gray)))

        fig = go.Figure(data=[
            go.Surface(
                z=z1,
                x=x,
                y=y,
                colorscale=colorscale,
                showscale=False,
                lighting=dict(
                    ambient=1.0,
                    diffuse=1.0
                )
            )
        ])
        for _x, _y, _zmin, _zmax in zip(x.flatten(), y.flatten(), z2.flatten(), z3.flatten()):
            fig.add_traces(
                go.Scatter3d(
                    x=(_x, _x),
                    y=(_y, _y),
                    z=(_zmin, _zmax),
                    mode='lines',
                    line=dict(
                        color='rgba(0, 0, 0, 0.15)',
                        width=3
                    )
                )
            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=-1.25, z=1.25)
        )

        fig.update_layout(
            font_family='Helvetica',
            title_font_family='Helvetica',
            title='%s vs. %s' % (get_irf_name(xvar, model.irf_name_map), get_irf_name(yvar, model.irf_name_map)),
            scene = dict(
                xaxis_title=get_irf_name(xvar, model.irf_name_map),
                yaxis_title=get_irf_name(yvar, model.irf_name_map),
                zaxis_title=get_irf_name(response, model.irf_name_map) + ", " + resparam,
                xaxis=dict(range=[xmin, xmax], gridcolor='rgb(200, 200, 200)', showbackground=False, autorange='reversed'),
                yaxis=dict(range=[ymin, ymax], gridcolor='rgb(200, 200, 200)', showbackground=False),
                zaxis=dict(range=[zmin, zmax], gridcolor='rgb(200, 200, 200)', showbackground=False)
            ),
            plot_bgcolor='rgb(255, 255, 255)',
            paper_bgcolor='rgb(255, 255, 255)',
            scene_camera=camera,
            showlegend=False
        )

    return fig

def layout():
    reference_settings = [
        html.Div(
            className='fullwidth-app-controls-name',
            children='Reference predictor values'
    )]
    for x in model.impulse_names:
        reference_settings.append(
            html.Label(
                id='%s-reference-label' % x,
                children=[
                    get_irf_name(x, model.irf_name_map),
                    dcc.Input(
                        id='%s-reference' % x,
                        type='number',
                        debounce=True,
                        placeholder=model.reference_arr[model.impulse_names_to_ix[x]],
                        value=model.reference_arr[model.impulse_names_to_ix[x]]
                    )
                ]
            )
        )

    return html.Div(
        id='cdrnn-body',
        className='app-body',
        children=[
            html.Div(
                id="viewport-wrapper",
                children=[
                    dcc.Loading(
                    type="dot",
                    fullscreen=False,
                    children=html.Div(
                        id='viewport',
                        children=dcc.Graph(
                            id='graph',
                            config=dict(
                                editable=True,
                            ),
                            style={'width': '70vw', 'height': '90vh'}
                        ),
                    ))
                ]
            ),
            html.Div(
                id='cdrnn-settings',
                className='control-settings',
                children=html.Div(
                    id='cdrnn-settings-inner',
                    children=[
                        html.Div(
                            title='Axis settings',
                            className='app-controls-block',
                            children=[
                                html.Div(
                                    className='fullwidth-app-controls-name',
                                    children='Axes'
                                ),
                                html.Label(
                                    children=[
                                        'X axis',
                                        dcc.Dropdown(
                                            id='dropdown_x',
                                            options=[{'label': get_irf_name(i, model.irf_name_map), 'value': i} for i in options],
                                            value=options[0],
                                            clearable=False
                                        )
                                    ]
                                ),
                                html.Label(
                                    children=[
                                        'Y axis (optional)',
                                        dcc.Dropdown(
                                            id='dropdown_y',
                                            options=[{'label': get_irf_name(i, model.irf_name_map), 'value': i} for i in options],
                                            value=options[options.index('t_delta')],
                                            clearable=True
                                        )
                                    ]
                                ),
                                html.Label(
                                    children=[
                                        'Response variable',
                                        dcc.Dropdown(
                                            id='dropdown_response',
                                            options=[{'label': get_irf_name(i, model.irf_name_map), 'value': i} for i in response_options],
                                            value=response_options[0],
                                            clearable=False
                                        )
                                    ]
                                ),
                                html.Label(
                                    children=[
                                        'Response parameter',
                                            dcc.Dropdown(
                                            id='dropdown_resparams',
                                            options=[{'label': x, 'value': x} for x in get_resparams(model, response_options[0])],
                                            value=model.expand_param_name(response_options[0], model.get_response_params(response_options[0])[0])[0],
                                            clearable=False
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            title='Reference values',
                            className='app-controls-block',
                            children=reference_settings
                        ),
                        html.Div(
                            title='Uncertainty',
                            className='app-controls-block',
                            children=[
                                html.Div(className='fullwidth-app-controls-name',
                                         children='Uncertainty'),
                                html.Label(
                                    children=[
                                        'Number of samples',
                                        dcc.Input(
                                            id='n_samples',
                                            type='number',
                                            debounce=True,
                                            placeholder='Number of samples',
                                            min=0,
                                            step=1,
                                            value=10
                                        )
                                    ]
                                ),
                                html.Label(
                                    children=[
                                        'Error interval, 0-100 (default: 95)',
                                        dcc.Input(
                                            id='ci',
                                            type='number',
                                            debounce=True,
                                            placeholder='Error interval, 0-100 (default: 95)',
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=95
                                        )
                                    ]
                                )
                            ]),
                        html.Div(
                            title='Axis bounds',
                            className='app-controls-block',
                            children=[
                                html.Div(className='fullwidth-app-controls-name',
                                         children=' Axis bounds'),

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
                    ]
                )
            )
        ]
    )

def callbacks(_app):
    args = [
        Output('graph', 'figure'),
        Input('dropdown_x', 'value'),
        Input('dropdown_response', 'value'),
        Input('dropdown_resparams', 'value'),
        Input('dropdown_y', 'value'),
        Input('n_samples', 'value'),
        Input('ci', 'value'),
        Input('x_min', 'value'),
        Input('x_max', 'value'),
        Input('y_min', 'value'),
        Input('y_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value'),
    ] + [
        Input('%s-reference' % x, 'value') for x in model.impulse_names
    ]
    @_app.callback(*args)
    def update_graph(
            xvar,
            yvar,
            response,
            resparam,
            n_samples,
            level,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            *args
    ):
        X_ref = {}
        for x, arg in zip(model.impulse_names, args):
            if arg is not None:
                X_ref[x] = arg
        return generate_figure(
            xvar,
            yvar,
            response,
            resparam,
            n_samples,
            level,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            X_ref=X_ref,
        )

    @_app.callback(
        Output('dropdown_resparams', 'options'),
        Input('dropdown_response', 'value')
    )
    def update_response_param_options(response_value):
        return [{'label': x, 'value': x} for x in get_resparams(model, response_value)]

    @_app.callback(
        Output('dropdown_resparams', 'value'),
        Input('dropdown_response', 'value')
    )
    def update_response_param_value(response_value):
        return model.get_response_params(response_value)[0]

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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("""
    Start a web server for interactive CDR visualization.
    """)
    argparser.add_argument('model', help='Path to model directory')
    argparser.add_argument('-d', '--debug', action='store_true', help='Whether to run in debug mode.')
    args = argparser.parse_args()

    model = model_generation(args.model)
    model.set_predict_mode(True)
    options = (model.impulse_names if model.has_nn_irf else model.impulse_names)[:]
    options += ['t_delta', 'X_time']
    response_options = model.response_names
    response = response_options[0]
    response_param = model.get_response_params(response)[0]
    response_param = model.expand_param_name(response, response_param)
    manipulations = [{model.impulse_names[0]: 1}]
    app = run_standalone_app(layout, callbacks, __file__)
    server = app.server
    app.run_server(debug=args.debug, port=5000)
