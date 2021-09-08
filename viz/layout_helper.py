import base64
import os

import dash
import dash_core_components as dcc
import dash_html_components as html


def run_standalone_app(
        layout,
        callbacks,
        header_colors,
        filename
):
    app = dash.Dash(__name__)
    app.scripts.config.serve_locally = True

    app.config['suppress_callback_exceptions'] = True

    app_name = 'CDR(NN) Visualization Tool'

    app_title = 'CDR(NN) Visualization Tool'

    app.layout = app_page_layout(
        page_layout=layout(),
        app_title=app_title,
        app_name=app_name,
        standalone=True,
        **header_colors()
    )

    callbacks(app)

    return app


def app_page_layout(page_layout,
                    app_title="CDR(NN) Visualization Tool",
                    app_name="CDR(NN) Visualization Tool",
                    light_logo=True,
                    standalone=False,
                    bg_color="#506784",
                    font_color="#F3F6FA"):
    return html.Div(
        id='main_page',
        children=[
            dcc.Location(id='url', refresh=False),
            html.Div(
                id='app-page-header',
                children=[
                    html.H2(
                        '......' + app_title
                    )
                ],
                style={
                    'background': bg_color,
                    'color': font_color,
                }
            ),
            html.Div(
                id='app-page-content',
                children=page_layout
            )
        ],
    )