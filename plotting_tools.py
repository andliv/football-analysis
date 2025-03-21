# PACKAGES
import Metrica_Viz as mviz
import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import Voronoi
import plotly.graph_objects as go

def plot_pitch(field_dimen = (106.0,68.0)):
    
    lc = 'black'
    pc = 'black'
    markersize = 8
    linewidth = 2
    border_dimen = (3,3) 
    meters_per_yard = 0.9144
    half_pitch_length = field_dimen[0]/2.
    half_pitch_width = field_dimen[1]/2.
    signs = [-1,1] 
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard

    fig = go.Figure()

    # Draw centerline
    fig.add_trace(go.Scatter(x = [0, 0], y = [-half_pitch_width, half_pitch_width], mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))

    # Draw center circle
    theta = np.linspace(0, 2**np.pi, 100)
    fig.add_trace(go.Scatter(x = np.cos(theta) * centre_circle_radius,
                            y = np.sin(theta) * centre_circle_radius,
                            mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"
                            ))
    
    # Center spot
    fig.add_trace(go.Scatter(x = [0], y = [0], mode = "markers", marker = dict(color = pc, size = markersize), showlegend=False, hoverinfo="skip"))

    for s in signs:
        # Pitch boundaries
        fig.add_trace(go.Scatter(x = [-half_pitch_length, half_pitch_length], y = [s*half_pitch_width, s*half_pitch_width], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x = [s*half_pitch_length, s*half_pitch_length], y = [-half_pitch_width, half_pitch_width], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))

        # Penalty area
        fig.add_trace(go.Scatter(x = [s*half_pitch_length, s*half_pitch_length-s*area_length], y = [area_width/2, area_width/2], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x = [s*half_pitch_length, s*half_pitch_length-s*area_length], y = [-area_width/2, -area_width/2], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x = [s*half_pitch_length - s*area_length, s*half_pitch_length - s*area_length], y = [-area_width/2, area_width/2], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))

        # Penalty spot
        fig.add_trace(go.Scatter(x = [s*half_pitch_length - s*penalty_spot], y = [0], mode = "markers", marker = dict(color = pc, size = markersize), showlegend=False, hoverinfo="skip"))

        # Goal area
        fig.add_trace(go.Scatter(x = [s*half_pitch_length, s*half_pitch_length-s*box_length], y = [box_width/2, box_width/2], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x = [s*half_pitch_length, s*half_pitch_length-s*box_length], y = [-box_width/2, -box_width/2], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x = [s*half_pitch_length - s*box_length, s*half_pitch_length - s*box_length], y = [-box_width/2, box_width/2], 
                                 mode = "lines", line = dict(color = lc, width = linewidth), showlegend=False, hoverinfo="skip"))

        # Goal posts
        fig.add_trace(go.Scatter(x = [s*half_pitch_length, s*half_pitch_length], y = [-goal_line_width/2, goal_line_width/2], 
                                 mode = "markers", marker = dict(color = pc, size = markersize), showlegend=False, hoverinfo="skip"))

        # Draw the D
        y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        fig.add_trace(go.Scatter(
            x=s * half_pitch_length - s * x,
            y=y,
            mode='lines',
            line=dict(color=lc, width=linewidth), hoverinfo="skip"
        ))

        # Corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)

        fig.add_trace(go.Scatter(
            x=s * half_pitch_length - s * x, 
            y=-half_pitch_width + y, 
            mode='lines', 
            line=dict(color=lc, width=linewidth), hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=s * half_pitch_length - s * x, 
            y=half_pitch_width - y, 
            mode='lines', 
            line=dict(color=lc, width=linewidth), hoverinfo="skip"
        ))

    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(range = [-half_pitch_length-3, half_pitch_length+3], visible = False),
        yaxis = dict(range = [-half_pitch_width-3, half_pitch_width+3], visible = False),
        showlegend = False,
        width = 1000,
        height = 600
    )
    return fig

def plot_frame(hometeam, awayteam, fig=None, team_colors=("blue", "red"), field_dimen = (106.0, 68.0), include_player_velocities = False, PlayerMarkerSize = 10, PlayerAlpha = 0.7, annotate=False):
    
    if fig is None: # create new pitch 
        fig = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig = fig # unpack tuple
    
    # plot home & away teams in order
    for team,color in zip( [hometeam,awayteam], team_colors) :
        x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
        y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
        fig.add_trace(go.Scatter(
            x = team[x_columns],
            y = team[y_columns],
            mode = "markers",
            marker=dict(color = color, size = PlayerMarkerSize), hoverinfo="skip"
            ))
        
        if include_player_velocities:
            # Get velocity columns (vx, vy) for each player
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns]  # column header for player x velocities
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns]  # column header for player y velocities
            
            for x_col, y_col, vx_col, vy_col in zip(x_columns, y_columns, vx_columns, vy_columns):
                x, y, vx, vy = team[x_col], team[y_col], team[vx_col], team[vy_col]
                fig.add_trace(go.Scatter(
                    x = [x, x+vx], y = [y, y + vy],
                    marker = dict(size = 5, symbol = "arrow-bar-up", angleref = "previous", color = color), hoverinfo="skip"
                    # mode = "lines",
                    # line = dict(color = color, width = 2), showlegend=False, hoverinfo="skip"
                ))
    
    fig.add_trace(go.Scatter(
        x = [hometeam["ball_x"]], y = [hometeam["ball_y"]], mode = "markers",
        marker = dict(size = 8, color = "black"), name = "Ball", hoverinfo="skip"
        ))
    
    return fig


def plot_voronoi(fig, event_id, events, tracking_home, tracking_away):
    """
    Adds Voronoi diagram to a Plotly figure based on player positions.
    """
    pass_frame = events.loc[event_id]['Start Frame']

    fig = plot_frame(tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], include_player_velocities = True)
    fig = plot_events(events.iloc[event_id], fig)
    # pass_team = events.loc[event_id].Team

    # Extract player positions for the frame
    home_frame_data = tracking_home.iloc[pass_frame]
    away_frame_data = tracking_away.iloc[pass_frame]
    
    # Home team positions
    home_x = home_frame_data.filter(like='Home').filter(like='_x').values
    home_y = home_frame_data.filter(like='Home').filter(like='_y').values
    home_positions = np.column_stack((home_x, home_y))
    home_colors = ['blue'] * len(home_positions)

    # Away team positions
    away_x = away_frame_data.filter(like='Away').filter(like='_x').values
    away_y = away_frame_data.filter(like='Away').filter(like='_y').values
    away_positions = np.column_stack((away_x, away_y))
    away_colors = ['red'] * len(away_positions)

    # Combine all positions and filter out NaN rows
    positions = np.vstack((home_positions, away_positions))
    colors = home_colors + away_colors
    
    # Filter out any NaN positions
    valid_indices = ~np.isnan(positions).any(axis=1)
    positions = positions[valid_indices]
    colors = [color for i, color in enumerate(colors) if valid_indices[i]]

    # Add boundary points to stabilize Voronoi regions
    positions = np.vstack((
        positions,
        [-70, -70], [70, 70], [70, -70], [-70, 70]  # Boundary points in the centered coordinate system
    ))

    # Calculate Voronoi regions
    vor = Voronoi(positions)
    
    # Define the pitch polygon
    pitch_polygon = ShapelyPolygon([(-53, -34), (53, -34), (53, 34), (-53, 34)])

    # Plot Voronoi regions
    for i, region_index in enumerate(vor.point_region[:-4]):  # Ignore the last 4 boundary regions
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 0:
            # Create a shapely polygon for the Voronoi region
            voronoi_polygon = ShapelyPolygon([vor.vertices[v] for v in region])

            # Get the intersection of the Voronoi polygon with the pitch polygon
            intersection_polygon = voronoi_polygon.intersection(pitch_polygon)

            # Plot the intersection of the Voronoi polygon with the pitch
            if not intersection_polygon.is_empty:
                color = colors[i]  # Use the color corresponding to the player
                x, y = intersection_polygon.exterior.xy
                fig.add_trace(go.Scatter(
                    x=list(x),
                    y=list(y),
                    fill='toself',
                    fillcolor=color,
                    opacity=0.3,
                    line=dict(color=color),
                    mode='lines',
                    hoverinfo="skip"
                ))
    
    return fig

def plot_events(event, fig=None, field_dimen = (106.0, 68), indicators = ["Marker", "Arrow"], color = "black", alpha = 0.5, annotate = False):
    
    if fig is None:
        fig = plot_pitch(field_dimen = field_dimen)

    if 'Marker' in indicators:
        fig.add_trace(go.Scatter(
            x = [event["Start X"]],
            y = [event["Start Y"]], 
            mode = "markers", 
            marker = dict(color = color, size = 8)
        ))

    if "Arrow" in indicators:
        fig.add_trace(go.Scatter(
            x = [event["Start X"], event["End X"]],
            y = [event["Start Y"], event["End Y"]],
            marker = dict(size = 6, symbol = "arrow-bar-up", angleref = "previous", color = color)
        ))

    if annotate:
        textstring = f"{event["Type"]}: {event["From"]}"
        fig.add_annotation(
            x = event["Start X"],
            y = event["Start Y"],
            text = textstring,
            font = dict(size = 10, color = color),
            opacity=alpha
        )

    return fig


def plot_pitchcontrol_for_event(event_id, events, tracking_home, tracking_away, PPCF, alpha = 0.7, include_player_velocities = True, annotate=False, field_dimen = (106.0, 68)):
    
    # pick a pass at which to generate the pitch control surface
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    
    # plot frame and event
    fig = plot_pitch(field_dimen = field_dimen)
    plot_frame(tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], fig = fig, PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate)
    plot_events( events.iloc[event_id], fig = fig, indicators = ['Marker','Arrow'], annotate=False, color= 'black', alpha=1 )

    if pass_team=='Home':
        custom_colorscale = [[0.0, "red"],
                            [0.5, "white"],
                            [1.0, "blue"]]
    else:
        custom_colorscale = [[0.0, "blue"],
                            [0.5, "white"],
                            [1.0, "red"]]

    # PPCF_flipped = np.flipud(PPCF)

    fig.add_trace(go.Contour(
        z=PPCF,  # Flipped PPCF matrix
        x=np.linspace(-field_dimen[0]/2., field_dimen[0]/2., PPCF.shape[1]),
        y=np.linspace(-field_dimen[1]/2., field_dimen[1]/2., PPCF.shape[0]),
        colorscale=custom_colorscale,
        zmin=0.0,
        zmax=1.0,
        opacity=0.6,
        showlegend=False,
        ncontours=100,
        showscale=False,
        contours=dict(showlines=False), hoverinfo="skip"
    ))
    
    return fig    

def plot_epv_for_event(event_id, events, tracking_home, tracking_away, PPCF, EPV, alpha = 0.7, include_player_velocities = True, annotate = False, autoscale = 0.05, field_dimen = (106.0, 68)):
    
    # pick a pass at which to generate the pitch control surface
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    print(pass_team)
    
    # plot frame and event
    fig = plot_pitch(field_dimen = field_dimen)
    plot_frame( tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], fig=fig, PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    plot_events( events.iloc[event_id], fig = fig, indicators = ['Marker','Arrow'], annotate=False, color= 'black', alpha=1 )
       
    # plot pitch control surface
    if pass_team=='Home':
        color = 'Blues'
        EPV = np.fliplr(EPV) if mio.find_playing_direction(tracking_home,'Home') == -1 else EPV
    else:
        color = 'Reds'
        EPV = np.fliplr(EPV) if mio.find_playing_direction(tracking_away,'Away') == -1 else EPV

    EPVxPPCF = PPCF*EPV
    
    if autoscale is True:
        vmax = np.max(EPVxPPCF)*2.
    elif autoscale>=0 and autoscale<=1:
        vmax = autoscale
    else:
        assert False, "'autoscale' must be either {True or between 0 and 1}"
        
    fig.add_trace(go.Contour(
        z=EPVxPPCF,  # Flipped PPCF matrix
        x=np.linspace(-field_dimen[0]/2., field_dimen[0]/2., EPVxPPCF.shape[1]),
        y=np.linspace(-field_dimen[1]/2., field_dimen[1]/2., EPVxPPCF.shape[0]),
        colorscale=color,
        zmin=0.0,
        zmax=vmax,
        opacity=0.7,
        showlegend=False,
        ncontours=100,
        showscale=False,
        contours=dict(showlines=False)
    ))
    
    return fig
