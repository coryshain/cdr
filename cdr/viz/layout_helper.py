import base64
import os

import dash
import dash_core_components as dcc
import dash_html_components as html


def run_standalone_app(
        layout,
        callbacks,
        filename
):
    app = dash.Dash(__name__)
    app.scripts.config.serve_locally = True

    app.config['suppress_callback_exceptions'] = True

    app_name = 'CDR Viewer'

    app_title = 'CDR Viewer'

    app.layout = app_page_layout(
        page_layout=layout(),
        app_title=app_title,
        app_name=app_name,
        standalone=True
    )

    callbacks(app)

    return app


def app_page_layout(page_layout,
                    app_title="CDR Viewer",
                    app_name="CDR Viewer",
                    light_logo=True,
                    standalone=False,
                    font_color="#FFFFFF"):
    return html.Div(
        id='main_page',
        children=[
            dcc.Location(id='url', refresh=False),
            html.Div(
                id='app-page-content',
                children=page_layout
            )
        ],
    )