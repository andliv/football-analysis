import Metrica_Viz as mviz
import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotting_tools import plot_pitch, plot_frame, plot_events, plot_pitchcontrol_for_event, plot_epv_for_event, plot_voronoi

import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_auth


DATADIR = "sample-data-master/data"

game_id = 2 # let's look at sample match 2

# READ IN EVENT DATA
events = mio.read_event_data(DATADIR, game_id)
events = mio.to_metric_coordinates(events)
events = events.round(2)
events = events.reset_index(names="event_id")
events = events[["event_id", "Type", "Subtype", "Team", "From", "To", "Period", "Start Time [s]",
                 "End Time [s]", "Start Frame", "End Frame", "Start X", "Start Y",
                 "End X", "End Y"]]

home_events = events[events['Team']=='Home']
away_events = events[events['Team']=='Away']

# READING IN TRACKING DATA
tracking_home = mio.tracking_data(DATADIR,game_id,'Home')
tracking_away = mio.tracking_data(DATADIR,game_id,'Away')

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

tracking_home = mvel.calc_player_velocities(tracking_home)
tracking_away = mvel.calc_player_velocities(tracking_away)

# READ IN EPV
epv = mepv.load_EPV_grid("EPV_grid.csv")

# READ MODEL PARAMS
params = mpc.default_model_params()

# GET GOALKEEPER NRS
gk_nrs = (mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away))

fig = plot_pitch()
fig = plot_frame(tracking_home.iloc[0], tracking_away.iloc[0], fig=fig, include_player_velocities=True)
fig.update_layout(
    autosize=True, height=500, width=900,  # Reduced height slightly
    margin=dict(l=0, r=0, t=20, b=20), paper_bgcolor="#f8f9fa"
)

VALID_USERNAME_PASSWORD_PAIRS = {
    'andlin': 'test123',
    'gugge': 'ikb25'
}
secret_key = "andlinsecret123!"


# Initialize Dash app with a modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS,
    secret_key=secret_key
)

app.layout = dbc.Container(fluid=True, children=[

    dbc.Row([
        dbc.Col(html.H1("Football Decision Evaluator",
                        className="text-center text-primary fw-bold",
                        style={"paddingTop": "15px"}  # Adds space above the title
                        ), width=12)
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col(html.P(
            "The table below contains publicly available event data from an anonymous game. "
            "Select an event/row to evaluate it in the plot. The event is evaluated based on Expected Possession Value, "
            "but you can also use a Voronoi diagram or Pitch Control Model.",
            className="lead text-dark text-center"
        ), width=12)
    ]),

    html.Br(),

    # Side-by-side Table & Graph (Adjusted Heights)
    dbc.Row([
        # Table Column (Shorter height)
        dbc.Col([
            dag.AgGrid(
                id="grid",
                rowData=events.to_dict("records"),
                columnDefs=[{"field": i} for i in events.columns],  # Show all columns
                dashGridOptions={"rowSelection": "single"},
                defaultColDef={"filter": "agTextColumnFilter"},
                style={"height": "350px", "width": "100%", "font-family": "Open Sans", "font-size": "14px"},
            ),
            html.Br(),  # Space before radio buttons
            dcc.RadioItems(
                id="radio-button",
                options=[
                    {"label": "Expected Possession Value", "value": "Expected Possession Value"},
                    {"label": "Pitch Control", "value": "Pitch Control"},
                    {"label": "Voronoi Diagram", "value": "Voronoi Diagram"}
                ],
                value="Expected Possession Value",
                inline=True,
                className="text-center",
                style={"font-family": "Open Sans"}
            )
        ], width=4),  # Table + Radio Buttons Column

        # Graph Column (Reduced height)
        dbc.Col([
            dcc.Graph(
                figure=fig, id="graph1",
                style={"width": "100%", "height": "500px"}  # Adjusted height
            )
        ], width=8)  # Graph Column
    ], className="mb-3")  # Adds spacing below
])


@app.callback(
    Output("graph1", "figure"),
    [Input("grid", "selectedRows"), Input("radio-button", "value")]
)
def update_graph(my_row, radio_btn):
    fig = plot_pitch()
    fig = plot_frame(tracking_home.iloc[0], tracking_away.iloc[0], fig=fig, include_player_velocities=True)
    fig.update_layout(
        autosize=True, height=500, width=900,  # Adjusted for full fit
        margin=dict(l=0, r=0, t=20, b=20), paper_bgcolor="#f8f9fa"
    )

    if my_row:
        event_id = my_row[0]["event_id"]

        if radio_btn.strip() == "Expected Possession Value":
            ppcf, xgrid, ygrid = mpc.generate_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, gk_nrs)
            fig = plot_epv_for_event(event_id, events, tracking_home, tracking_away, ppcf, epv)
        elif radio_btn.strip() == "Pitch Control":
            ppcf, xgrid, ygrid = mpc.generate_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, gk_nrs)
            fig = plot_pitchcontrol_for_event(event_id, events, tracking_home, tracking_away, ppcf)
        else:
            fig = plot_voronoi(fig, event_id, events, tracking_home, tracking_away)

        fig.update_layout(
            autosize=True, height=500, width=900,  # Adjusted to fit without scrolling
            margin=dict(l=0, r=0, t=20, b=20), paper_bgcolor="#f8f9fa"
        )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)