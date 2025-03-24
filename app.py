import Metrica_Viz as mviz
import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotting_tools import plot_pitch, plot_frame, plot_events, plot_pitchcontrol_for_event, plot_epv_for_event, plot_voronoi, plot_max_val_added
import pickle
import gdown

import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_auth


file_id_events = "1rMsMlS-sipbvbk1WOCS2qUs4P3Soaoej"
file_id_tracking_home = "1DZstm5Lq_Yqm5qJpgAoarhRzEarejrGF"
file_id_tracking_away = "1CRL1Uwn53ITwMHb8mfWb4lpxOgdbusRF"
file_id_pitch_control = "1Dd-82tYIDPd3Rahgxr7pVcIdfGbcVnvi"
file_id_max_val = "1tqw2DpKvfuaVplt_V9unfcPSsS2clMJV"

# READ IN EVENT DATA
events = mio.read_event_data(file_id_events)
events = mio.to_metric_coordinates(events)
events = events.round(2)
events = events.reset_index(names="event_id")
events = events[["event_id", "Type", "Subtype", "Team", "From", "To", "Period", "Start Time [s]",
                 "End Time [s]", "Start Frame", "End Frame", "Start X", "Start Y",
                 "End X", "End Y"]]

home_events = events[events['Team']=='Home']
away_events = events[events['Team']=='Away']

# READING IN TRACKING DATA
tracking_home = mio.tracking_data(file_id_tracking_home, "Home")
tracking_away = mio.tracking_data(file_id_tracking_away, "Away")

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

tracking_home = mvel.calc_player_velocities(tracking_home)
tracking_away = mvel.calc_player_velocities(tracking_away)

# READ IN EPV
# epv_path = Path(__file__).parent.parent/"data"/"EPV_grid.csv"
epv = mepv.load_EPV_grid("EPV_grid.csv")

# READ MODEL PARAMS
params = mpc.default_model_params()

# GET GOALKEEPER NRS
gk_nrs = (mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away))

# READ CACHED DATA
# pitch_control_cache_path = Path(__file__).parent.parent/"data"/"pitch_control_cache.pkl"
output = "data.pkl"
gdown.download(f"https://drive.google.com/uc?id={file_id_pitch_control}", output, quiet=False)
with open(output, "rb") as f:
    pitch_control_cache_r = pickle.load(f)

# max_val_added_cache_path = Path(__file__).parent.parent/"data"/"max_val_added_cache.pkl"
output = "data.pkl"
gdown.download(f"https://drive.google.com/uc?id={file_id_max_val}", output, quiet=False)
with open(output, "rb") as f:
    max_val_added_cache_r = pickle.load(f)

fig = plot_pitch()
fig = plot_frame(tracking_home.iloc[0], tracking_away.iloc[0], fig=fig, include_player_velocities=True)
fig.update_layout(
    autosize=True, height=500, width=900,
    margin=dict(l=0, r=0, t=20, b=20), paper_bgcolor="#f8f9fa"
)

VALID_USERNAME_PASSWORD_PAIRS = {
    'andlin': 'test123',
    'gugge': 'ikb25'
}
secret_key = "andlinsecret123!"


# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
server = app.server
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS,
    secret_key=secret_key
)

app.layout = dbc.Container(fluid=True, children=[

    dbc.Row([
        dbc.Col(html.H1("Football Decision Evaluator",
                        className="text-center text-primary fw-bold",
                        style={"paddingTop": "15px"}
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

    dbc.Row([
        dbc.Col([
            dag.AgGrid(
                id="grid",
                rowData=events.to_dict("records"),
                columnDefs=[{"field": i} for i in events.columns],
                dashGridOptions={"rowSelection": "single"},
                defaultColDef={"filter": "agTextColumnFilter"},
                style={"height": "350px", "width": "100%", "font-family": "Open Sans", "font-size": "14px"},
            ),
            html.Br(),
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
        ], width=4),

        dbc.Col([
            dcc.Graph(
                figure=fig, id="graph1",
                style={"width": "100%", "height": "500px"}
            )
        ], width=8)
    ], className="mb-3")
])


@app.callback(
    Output("graph1", "figure"),
    [Input("grid", "selectedRows"), Input("radio-button", "value")]
)
def update_graph(my_row, radio_btn):
    fig = plot_pitch()
    fig = plot_frame(tracking_home.iloc[0], tracking_away.iloc[0], fig=fig, include_player_velocities=True)
    fig.update_layout(
        autosize=True, height=500, width=900,
        margin=dict(l=0, r=0, t=20, b=20), paper_bgcolor="#f8f9fa"
    )

    if my_row:
        event_id = my_row[0]["event_id"]

        if radio_btn.strip() == "Expected Possession Value":
            fig = plot_epv_for_event(event_id, events, tracking_home, tracking_away, pitch_control_cache_r.get(event_id), epv)

            if event_id in max_val_added_cache_r:
                fig = plot_max_val_added(fig, event_id, max_val_added_cache_r)
            else:
                print(f"Skipping event {event_id} as it is not in max_val_added_cache_r")
            
        elif radio_btn.strip() == "Pitch Control":
            fig = plot_pitchcontrol_for_event(event_id, events, tracking_home, tracking_away, pitch_control_cache_r.get(event_id))
            
            if event_id in max_val_added_cache_r:
                fig = plot_max_val_added(fig, event_id, max_val_added_cache_r)
            else:
                print(f"Skipping event {event_id} as it is not in max_val_added_cache_r")
        
        else:
            fig = plot_voronoi(fig, event_id, events, tracking_home, tracking_away)
            if event_id in max_val_added_cache_r:
                fig = plot_max_val_added(fig, event_id, max_val_added_cache_r)
            else:
                print(f"Skipping event {event_id} as it is not in max_val_added_cache_r")
        
        fig.update_layout(
            autosize=True, height=500, width=900,
            margin=dict(l=0, r=0, t=20, b=20), paper_bgcolor="#f8f9fa"
        )

    return fig


if __name__ == "__main__":
    app.run_server(debug = True)
