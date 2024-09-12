import streamlit as st
import pandas as pd
from fitparse import FitFile
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
import gpxpy


@st.cache_data
def read_fit_file(fit_file_path):
    # Load the .fit file
    fitfile = FitFile(fit_file_path)

    # Extract data from the file
    data = {
        'time': [],
        'power': [],
        'altitude': [],
        'distance': [],
        'heart_rate': [],
        'latitude': [],
        'longitude': []
    }

    for record in fitfile.get_messages('record'):
        power = None
        timestamp = None
        altitude = None
        distance = None
        heart_rate = None
        lat = None
        long = None

        for data_point in record:
            if data_point.name == 'power':
                power = data_point.value
            if data_point.name == 'timestamp':
                timestamp = data_point.value
            if data_point.name == 'altitude':
                altitude = data_point.value
            if data_point.name == 'distance':
                distance = data_point.value
            if data_point.name == 'heart_rate':
                heart_rate = data_point.value
            if data_point.name == 'position_lat':
                lat = data_point.value
            if data_point.name == 'position_long':
                long = data_point.value

        # List of values to check for None
        values_to_check = [power, timestamp, altitude, distance, heart_rate, lat, long]

        if all(value is not None for value in values_to_check):
            data['power'].append(power)
            data['time'].append(timestamp)
            data['altitude'].append(altitude)
            data['distance'].append(distance)
            data['heart_rate'].append(heart_rate)
            data['latitude'].append(lat / (2 ** 32 / 360))
            data['longitude'].append(long / (2 ** 32 / 360))

    # Convert data to DataFrame
    df = pd.DataFrame(data)
    # Convert time to seconds
    df['time'] = pd.to_datetime(df['time'])
    df['time_sec'] = (df['time'] - df['time'].min()).dt.total_seconds()
    df['time_min'] = df['time_sec'] / 60
    df['time_h'] = df['time_min'] / 60
    df['distance_km'] = df['distance'] / 60

    # drop the time column
    df.drop(columns='time', inplace=True)
    return df


@st.cache_data
def read_gpx_file(uploaded_file):
    # Load the GPX file
    # with open(gpx_file_path, 'r') as gpx_file:
    #     gpx = gpxpy.parse(gpx_file)
    gpx = gpxpy.parse(uploaded_file.read().decode("utf-8"))

    # Extract data from the file
    data = {
        'time': [],
        'latitude': [],
        'longitude': [],
        'altitude': [],
        'power': [],
        'heart_rate': []
    }

    # Loop through all track points
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data['time'].append(point.time)
                data['latitude'].append(point.latitude)
                data['longitude'].append(point.longitude)
                data['altitude'].append(point.elevation)

                # Check for extensions for power and heart rate
                power = None
                heart_rate = None

                if point.extensions:
                    for ext in point.extensions:
                        if 'power' in ext.tag.lower():
                            power = int(ext.text) if ext.text.isdigit() else None
                        elif 'heartrate' in ext.tag.lower():
                            heart_rate = int(ext.text) if ext.text.isdigit() else None

                data['power'].append(power)
                data['heart_rate'].append(heart_rate)

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Handle time conversion
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time_sec'] = (df['time'] - df['time'].min()).dt.total_seconds()
        df['time_min'] = df['time_sec'] / 60
        df['time_h'] = df['time_min'] / 60

    return df


@st.cache_data
def get_reference_power_data(gender, unit_sel):
    power_df = pd.read_csv(f'data/reference_power_curve_{gender}.csv')
    power_unit = power_df.copy()
    if unit_sel == 'Watts':
        power_unit.drop(columns=power_unit.columns[range(6, 11)], inplace=True)
    else:
        power_unit.drop(columns=power_unit.columns[range(1, 6)], inplace=True)

    power_unit.set_index(power_unit.columns[0], inplace=True)
    power_unit_df = power_unit.transpose()

    description = ["5 sec", "1 min", "5 min", "20 min", "60 min"]
    seconds = [5, 60, 300, 1200, 3600]

    power_unit_df['description'] = description
    power_unit_df['seconds'] = seconds

    # Desired full list of data points
    full_data = [
        {"seconds": 5, "description": "5 sec"},
        {"seconds": 10, "description": "10 sec"},
        {"seconds": 30, "description": "30 sec"},
        {"seconds": 60, "description": "1 min"},
        {"seconds": 300, "description": "5 min"},
        {"seconds": 480, "description": "8 min"},
        {"seconds": 720, "description": "12 min"},
        {"seconds": 1200, "description": "20 min"},
        {"seconds": 2400, "description": "40 min"},
        {"seconds": 3600, "description": "60 min"}
    ]

    # Create a DataFrame for the full list of data points
    full_df = pd.DataFrame(full_data)

    # Merge the existing data with the full list
    merged_df = pd.merge(full_df, power_unit_df, on=['seconds', 'description'], how='left')

    return merged_df


def format_time(hour):
    hours = int(hour)  # Get the hour part
    minutes = int((hour - hours) * 60)  # Convert the decimal part to minutes
    seconds = int(((hour - hours) * 60 - minutes) * 60)  # Convert the remaining decimal part to seconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@st.cache_data
def get_interpolated_power_curve(df):
    # Create the interpolation function
    interpolator = interp1d(df['Watts'], df['seconds'], kind='linear', fill_value="extrapolate")

    # Generate interpolated values for each Watt between the first and last entry
    watts_range = np.arange(df['Watts'].min() * 0.8, df['Watts'].max() + 1)

    # Interpolate seconds for each Watt
    interpolated_seconds = interpolator(watts_range)

    # Create a new DataFrame with the interpolated values
    interpolated_df = pd.DataFrame({
        'Watts': watts_range,
        'Seconds': interpolated_seconds
    })

    interpolated_df['drain'] = 1 / interpolated_df['Seconds'] * 100

    return interpolated_df


@st.cache_data
def get_battery(df, interpolated_df):
    battery = [200]

    for val in df['power']:
        if val < interpolated_df['Watts'].min():
            battery.append(battery[-1])
        else:
            battery_drain = interpolated_df.loc[interpolated_df['Watts'] == int(val), 'drain'].values[0]
            battery.append(battery[-1] - battery_drain)

    # Remove the last value
    return battery[:-1]


def get_battery_color(level):
    if level > 75:
        return "green"
    elif level > 50:
        return "yellowgreen"
    elif level > 25:
        return "orange"
    else:
        return "red"


def check_watts_decreasing(df):
    # Sort the DataFrame by the 'seconds' column
    sorted_df = df.sort_values(by='seconds')

    # Check if 'Watts' is monotonically decreasing
    is_monotonically_decreasing = sorted_df['Watts'].is_monotonic_decreasing

    return is_monotonically_decreasing


# Streamlit app starts here
st.title('What happens to your energy levels while you cycle?')

st.info('We recommend the use of the Light colour scheme. This can be changed in the upper right corner.', icon="ℹ️")


st.image("data/cycling_cover.JPG")

st.write('This tool helps athletes better understand their '
        'endurance rides, which focus more on continuous individual effort than on tactics.')

# File uploader
uploaded_fit_file = st.sidebar.file_uploader("**Upload your ride**", type=["fit","gpx"])

# Power curve ----
st.subheader('Power curve')
input_df = pd.DataFrame(
    [
        {"seconds": 5, "description": "5 sec", "Watts": 0},
        {"seconds": 10, "description": "10 sec", "Watts": 0},
        {"seconds": 30, "description": "30 sec", "Watts": 0},
        {"seconds": 60, "description": "1 min", "Watts": 0},
        {"seconds": 300, "description": "5 min", "Watts": 0},
        {"seconds": 480, "description": "8 min", "Watts": 0},
        {"seconds": 720, "description": "12 min", "Watts": 0},
        {"seconds": 1200, "description": "20 min", "Watts": 0},
        {"seconds": 2400, "description": "40 min", "Watts": 0},
        {"seconds": 3600, "description": "60 min", "Watts": 0}
    ]
)

# Create two columns in Streamlit layout
col1, col2 = st.columns([1.5, 3], vertical_alignment='top')

col2.markdown("**Upload a previously exported CSV**")

# File upload section
uploaded_power_curve = col2.file_uploader("Upload your power curve CSV", type="csv")
if uploaded_power_curve is not None:
    # Read the uploaded file into a DataFrame
    input_df = pd.read_csv(uploaded_power_curve)

# Column for data input
col1.markdown("**Enter your data in the Watts columns**")
# Data editor for user input
edited_df = col1.data_editor(
    input_df,
    column_config={
        "seconds": None,
        "Watts": st.column_config.NumberColumn(
            "Watts",
            help="maximum Watts you can sustain for the given time period",
            min_value=1,
            step=1
        )
    },
    disabled=["seconds", "description"],
    hide_index=True,
)

# Download button for edited data
col2.write("If you have entered your data in the table, you can now download it as a csv file "
           "for later use.")
col2.download_button(
    label="Download CSV",
    data=edited_df.to_csv(index=False).encode('utf-8'),
    file_name=f'power_curve_{datetime.now().strftime("%Y-%m-%d")}.csv',
    mime='text/csv',
)

col2.markdown('**Athlete details**')
with col2:
    nested_col1, nested_col2 = st.columns(2)

# weight input
bodymass = nested_col1.number_input('Enter your weight in kg:', min_value=0, placeholder="kg", )

# gender
gender = nested_col2.radio(
    "What's your gender",
    ["female", "male"],
)

power_df = edited_df.copy()

# Update 'W_kg' column based on 'Watts' and 'bodymass'
if bodymass > 0:  # Ensure bodymass is not zero to avoid division by zero
    power_df['Watts/kg'] = power_df['Watts'] / bodymass
else:
    power_df['Watts/kg'] = 0

# Radio button to select Y-axis data if bodymass > 0
if bodymass > 0:
    y_axis_selection = st.radio(
        "",
        ('Watts', 'Watts/kg'),
        index=0,  # Default to 'Watts'
        horizontal=True
    )
else:
    y_axis_selection = 'Watts'  # Default to 'Watts' if bodymass is not entered

if not check_watts_decreasing(power_df):
    st.warning('Are you sure your power values are correct?'
               '\nTypically your power should decrease with increasing time duration.',
               icon="⚠️")
    
reference_power = get_reference_power_data(gender, y_axis_selection)

# Create the Plotly figure
fig = go.Figure()

# add reference data
# add percentiles
percentiles = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
# Define color scheme for traces
colors = [
    '#d94801',  # Red-Orange
    '#f16913',  # Dark Orange
    '#fd8d3c',  # Orange
    '#fdd0a2',  # Light Orange
    '#c6dbef',  # Pale Blue
    '#9ecae1',  # Very Light Blue
    '#6baed6',  # Light Blue
    '#4292c6',  # Light Medium Blue
    '#2171b5',  # Medium Blue
    '#08519c',  # Medium Dark Blue
    '#08306b'  # Dark Blue
]

# Add multiple traces to the figure
for i, perc in enumerate(percentiles):
    fig.add_trace(go.Scatter(
        x=reference_power['description'],  # X-axis based on 'description'
        y=reference_power[perc],  # Y-axis based on the selected column
        mode='lines+markers',  # Line with markers for interactivity
        name=f'{perc}th',  # Name of the trace for the legend
        line=dict(color=colors[i], width=1),  # Line styling with specified color
        marker=dict(size=2),  # Marker styling
        hoverinfo='text',  # Hover information
        hovertext=[f"{perc}th: {value:.1f} {y_axis_selection}" for value in reference_power[perc]],
        connectgaps=True
    ))

# Add the athletes trace
fig.add_trace(go.Scatter(
    x=power_df['description'],  # X-axis based on 'description'
    y=power_df[y_axis_selection],  # Y-axis based on the selected column ('Watts' or 'W_kg')
    mode='lines+markers',  # Line with markers for interactivity
    name='your data',  # Name of the trace for the legend
    line=dict(color='black', width=3),  # Line styling
    marker=dict(size=8),  # Marker styling
    hoverinfo='text',  # Hover information
    hovertext=[f"you: {value:.1f} {y_axis_selection}" for value in power_df[y_axis_selection]]
))

# Customize layout
fig.update_layout(
    title="Power Curve",
    legend=dict(
        title=dict(text='Percentiles and your data')  # Add legend title here
    ),
    xaxis_title="Time Duration",
    yaxis_title=y_axis_selection,
    width=600,
    height=500,
    hovermode='x unified',
    template='plotly_white'  # Plotly template for better aesthetics
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
st.caption("The black line shows your power curve, while colored lines refers to the percentile "
           "of riders of the selected gender based on the data from David Johnstone: "
           "https://www.cyclinganalytics.com/blog/2018/06/how-does-your-cycling-power-output-compare")

# battery section -------------------------------------------
st.subheader("Your energy level")

if uploaded_fit_file is not None and (power_df['Watts'] > 0).all():
    # Read data from the uploaded .fit file
    # st.write(uploaded_fit_file)

    with st.spinner('File is loading'):
        # df_raw = read_fit_file(uploaded_fit_file)
        if "fit" in uploaded_fit_file.name:
            df_raw = read_fit_file(uploaded_fit_file)
        else:
            df_raw = read_gpx_file(uploaded_fit_file)

    # smoothing slider
    smoothing = st.sidebar.slider('smoothing window for power (sec):', 1, 15, 5)

    df = df_raw.copy()  # Create a copy to avoid modifying the original DataFrame

    # Apply rolling mean only to the 'power' column
    df['power'] = df_raw['power'].rolling(window=smoothing, min_periods=1).mean()

    # Format the time
    df['formatted_time'] = df['time_h'].apply(format_time)

    st.session_state.time_range_display = (
        df['formatted_time'].iloc[0], df['formatted_time'].iloc[-1])

    # show the gps on the map
    map_data = df[['latitude', 'longitude']]
    st.sidebar.map(map_data)

    # battery
    interpolated_df = get_interpolated_power_curve(power_df)

    battery = get_battery(df, interpolated_df)

    df['battery'] = pd.Series(battery) / 2

    if df['battery'].min() < 0:
        battery_scale = df['battery'].min()
    else:
        battery_scale = 0

    # Create a figure with secondary y-axis
    battery_fig = go.Figure()

    # Power trace
    battery_fig.add_trace(
        go.Scatter(
            x=df['formatted_time'],
            y=df['power'],
            mode='lines',
            line=dict(color='black', width=2),
            name='Power',
            hoverinfo='text',  # Hover information
            hovertext=[f"{value:.1f} W" for value in df['power']]
        )
    )

    battery_fig.add_trace(
        go.Scatter(
            x=df['formatted_time'],
            y=df['battery'],
            mode='markers',
            marker=dict(
                color=df['battery'],  # Use battery values to color the line
                colorscale='RdYlGn',  # Color scale from red (low) to green (high)
                colorbar=dict(title="Battery (%)"),
                cmin=0,  # Minimum value for scaling
                cmax=100  # Maximum value for scaling
            ),
            name='Battery',
            yaxis='y2',
            hoverinfo='text',  # Hover information
            hovertext=[f"{value:.1f}%" for value in df['battery']],
            showlegend=True
        )
    )

    ftp_value = power_df.loc[power_df['seconds'] == 3600, 'Watts'].values[0]

    battery_fig.add_shape(
        type="line",
        x0=df['formatted_time'].min(),  # Start from the minimum time
        x1=df['formatted_time'].max(),  # End at the maximum time
        y0=ftp_value,  # y position at ftp_value
        y1=ftp_value,  # y position at ftp_value
        line=dict(color="grey", width=2, dash="dash"),  # Line properties
        name="FTP",
        showlegend=True
    )

    battery_fig.update_xaxes(nticks=8)

    battery_fig.update_layout(
        xaxis=dict(title='Time (hours)'),
        yaxis=dict(
            title='Power (W)',
            showgrid=False,
            zeroline=False,
            range=[0, df['power'].max()]
        ),
        yaxis2=dict(
            overlaying='y',  # Overlay on the same plot
            side='right',  # Position on the right side
            position=0.95,  # Adjust position to avoid overlap with y2
            showgrid=False,
            zeroline=False,
            range=[battery_scale, 105]
            # showticklabels=False
            # tickfont=dict(color='blue')
        ),
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,  # Center horizontally
            xanchor="center",  # Align center
            y=1.0,  # Position above the plot
            yanchor="bottom"  # Anchor to bottom
        ),
        showlegend=True,
        hovermode='x unified',
        # width=600,
        # height=400,
    )

    # Display the plot in Streamlit
    st.plotly_chart(battery_fig)

    ##### battery schema

    col1, col2 = st.columns([1, 1], vertical_alignment="center")

    col1.markdown(
        "<p style='font-size:20px; font-weight:bold;'>Battery level at the end of the ride:</p>",
        unsafe_allow_html=True
    )

    battery_level = df.iloc[-1]['battery']
    if battery_level <3:
        battery_level_axis = 3
    else:
        battery_level_axis = battery_level

    # Create figure for battery
    fig = go.Figure()

    # Add the empty battery shell (background)
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=100, y1=1,
        line=dict(color="black", width=3),
        fillcolor="white"
    )

    # Add the filled battery level
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=battery_level_axis, y1=1,
        fillcolor=get_battery_color(battery_level),
        line=dict(color="black", width=0)
    )

    # Add battery "cap" at the end
    fig.add_shape(
        type="rect",
        x0=102, y0=0.2,
        x1=105, y1=0.8,
        line=dict(color="black", width=3),
        fillcolor="black"
    )

    # Add annotation for battery level percentage
    fig.add_annotation(
        x=5,  # Center of the filled battery level
        y=0.5,  # Middle of the battery height
        text=f"{battery_level:.1f}%",  # Text showing battery level
        showarrow=False,
        font=dict(size=24, color="black"),  # Adjust font size and color
        xanchor='left',  # Center alignment
        yanchor='middle'  # Middle alignment
    )

    # Update layout to remove gridlines and axes
    fig.update_layout(
        xaxis=dict(
            range=[-5, 110],  # Adjust range to show the cap
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-0.5, 1.5],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        width=200,  # Width of the figure
        height=100,  # Height of the figure
        margin=dict(l=0, r=0, t=0, b=0),  # Margins
        plot_bgcolor='white'  # Background color
    )

    # Display the plot in Streamlit
    col2.plotly_chart(fig)

    ### FTP section ----
    st.subheader("Your ride: power and a selected FTP treshhold")
    st.caption("FTP is a cycling metric that stands for Functional Threshold Power. "
               "It represents an approximation of your maximal lactate steady state, "
               "measured in watts.")

    col1, col2 = st.columns([1, 2.5], vertical_alignment="center", gap='medium')

    # FTP input
    ftp = col1.number_input('Enter your FTP:', min_value=0, placeholder="Type a number...",
                            step=1, value=power_df.loc[power_df['seconds'] == 3600, 'Watts'].values[0],
                            help="Manually adjust your FTP, if you think it does not correspond to the 60 min power entered above.")

    # FTP percentage input
    ftp_perc = col2.slider('Choose the percentage of FTP', 1, 125, 80)

    # selected ftp treshhold
    ftp_sel = ftp * ftp_perc / 100

    # Create two columns in Streamlit layout
    col1, col2 = st.columns([4, 1], gap="medium", vertical_alignment="center")

    # Initialize the session state for the slider
    if 'time_range_display' not in st.session_state:
        st.session_state.time_range_display = (
            df['formatted_time'].iloc[0], df['formatted_time'].iloc[-1])  # Default range

    # Button to reset the slider in the second column
    if col2.button('Reset time'):
        st.session_state.time_range_display = (df['formatted_time'].iloc[0], df['formatted_time'].iloc[-1])

    # Select Slider in the first column
    time_range_display = col1.select_slider(
        'Select a time range',
        options=df['formatted_time'].tolist(),  # Use formatted time for the slider display
        value=st.session_state.time_range_display  # Default to the stored range
    )

    # Update the session state with the new slider value if changed by the user
    if time_range_display != st.session_state.time_range_display:
        st.session_state.time_range_display = time_range_display

    # Convert selected time back to the float format for filtering
    start_time = df.loc[df['formatted_time'] == time_range_display[0], 'time_h'].iloc[0]
    end_time = df.loc[df['formatted_time'] == time_range_display[1], 'time_h'].iloc[0]

    # Filter DataFrame based on the selected time range
    filtered_df = df[(df['time_h'] >= start_time) & (df['time_h'] <= end_time)]

    # below and above the threshold values
    below_threshold = filtered_df[filtered_df['power'] <= ftp_sel]
    above_threshold = filtered_df[filtered_df['power'] > ftp_sel]

    # Create two columns in Streamlit layout
    col_plot, col_metric = st.columns([4, 1], vertical_alignment="center")

    # Create the plot using Plotly
    ftp_fig = go.Figure()

    # Add a light grey filled area for elevation in the background
    ftp_fig.add_trace(
        go.Scatter(
            x=filtered_df['formatted_time'],
            y=filtered_df['altitude'],
            mode='lines',
            line=dict(color='lightgrey', width=0),
            fill='tozeroy',  # Fill the area to the y=0 axis,
            yaxis='y',  # Use secondary y-axis
            name='Elevation'
        )
    )

    # Power trace
    ftp_fig.add_trace(
        go.Scatter(
            x=filtered_df['formatted_time'],
            y=filtered_df['power'],
            mode='lines',
            line=dict(color='black', width=2),
            name='Power',
            yaxis='y2'
        )
    )

    # Add a red transparent box using Scatter to make it appear in the legend without red points
    # Convert time_h min and max to formatted time
    x_min_formatted = format_time(filtered_df['time_h'].min())
    x_max_formatted = format_time(filtered_df['time_h'].max())

    # Add the trace with formatted x-axis values
    ftp_fig.add_trace(
        go.Scatter(
            x=[x_min_formatted, x_min_formatted, x_max_formatted, x_max_formatted, x_min_formatted],
            # Formatted time for x-axis
            y=[ftp_sel, filtered_df['power'].max(), filtered_df['power'].max(), ftp_sel, ftp_sel],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',  # Transparent orange
            line=dict(color='rgba(255, 0, 0, 0)'),  # No line around the filled area
            mode='lines',  # Set to 'lines' to remove markers
            name='Above FTP Threshold',  # Name for the legend
            showlegend=True,
            yaxis='y2'  # Assuming you have a secondary y-axis setup
        )
    )

    if filtered_df['heart_rate'].notna().all():
        # Add a trace for the heart rate using a third y-axis
        ftp_fig.add_trace(
            go.Scatter(
                x=filtered_df['formatted_time'],
                y=filtered_df['heart_rate'],
                mode='lines',
                line=dict(color='teal', width=2),
                name='Heart Rate',
                yaxis='y3'  # Use third y-axis
            )
        )

    ftp_fig.update_xaxes(nticks=6)

    # Update layout for the plot with a third y-axis
    ftp_fig.update_layout(
        # title='Power, Elevation, and Heart Rate Over Time',
        xaxis=dict(title='Time (hours)'),
        yaxis2=dict(
            title='Power (W)',
            overlaying='y',  # Overlay the secondary y-axis on the same plot
            side='left',
            showgrid=False,
            zeroline=False,
            range=[0, filtered_df['power'].max()]
        ),
        yaxis=dict(showticklabels=False),
        yaxis3=dict(
            title='Heart Rate (bpm)',
            overlaying='y',  # Overlay on the same plot
            side='right',  # Position on the right side
            anchor='free',  # Do not overlap with y2
            position=1,  # Adjust position to avoid overlap with y2
            showgrid=False,
            zeroline=False,
            tickfont=dict(color='teal')
        ),
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,  # Center horizontally
            xanchor="center",  # Align center
            y=1.0,  # Position above the plot
            yanchor="bottom"  # Anchor to bottom
        ),
        showlegend=True,
        hovermode='x unified',
        # width=600,
        # height=400,
    )

    # Display the interactive plot
    col_plot.plotly_chart(ftp_fig, use_container_width=True, on_select="rerun")

    # metric above/below Treshhold
    col1, col2 = st.columns(2)

    with col_metric:
        st.markdown(
            f"""
                <div style='text-align: center;'>
                    <p style='margin: 0;'>above treshhold</p>
                    <h2 style='display: inline; font-size: 2.5em;'> 
                        <span style='color: red;'>{above_threshold.shape[0] / filtered_df.shape[0] * 100:.1f}%</span>
                    </h2>
                </div>
                """,
            unsafe_allow_html=True
        )

    # Display "below" metric with colored percentage
    with col_metric:
        st.markdown(
            f"""
                <div style='text-align: center;'>
                    <p style='margin: 0;'>below treshhold</p>
                    <h2 style='display: inline; font-size: 2.5em;'> 
                        <span style='color: green;'>{below_threshold.shape[0] / filtered_df.shape[0] * 100:.1f}%</span>
                    </h2>
                </div>
                """,
            unsafe_allow_html=True
        )

elif not (power_df['Watts'] > 0).all():
    st.write('Please enter the power curve data above.')
elif uploaded_fit_file is None:
    st.write("Please upload a FIT file (on the left side). Afterwards you can proceed with the analysis of your ride.")

# farbige
