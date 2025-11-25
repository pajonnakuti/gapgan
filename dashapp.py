import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import base64
import io
import os
import sys
import glob
import uuid
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

# --- CONFIGURATION (Relative Paths for Git Portability) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Model Data Path
BASE_DATA_DIR = os.path.join(CURRENT_DIR, "model_data")

# 2. Coastline Path
COAST_SHAPEFILE_PATH = os.path.join(CURRENT_DIR, "assets", "indiacoast.shp")

# 3. DINCAE Path
DINCAE_PATH = os.path.join(CURRENT_DIR, "DINCAE")
if DINCAE_PATH not in sys.path:
    sys.path.append(os.path.dirname(DINCAE_PATH)) # Append parent so we can import DINCAE

try:
    import DINCAE
except ImportError:
    print(f"CRITICAL ERROR: Could not import DINCAE. Please check path: '{DINCAE_PATH}'")
    pass

HISTORICAL_ARCHIVE_NC_NAME = "latest.nc"
MODEL_CHECKPOINTS_SUBDIR = "checkpoints"
MODEL_CHECKPOINT_PATTERN = "model-*.ckpt.index"
VAR_NAME = "SST"
MISSING_VALUE = -999
TEMP_DIR = "temp_dash_files"

os.makedirs(TEMP_DIR, exist_ok=True)

# --- SECTOR BOUNDARIES ---
SECTOR_BOUNDARIES = {
    "1": {"name": "1_Gujarat",         "bounds": [[20.04315567, 67.88010406], [23.75766563, 72.97928619]]},
    "2": {"name": "2_Maharashtra",     "bounds": [[15.70783329, 70.50718689], [20.04315567, 73.71035767]]}, 
    "3": {"name": "3_Goa",             "bounds": [[14.85914707, 71.58411407], [15.70611858, 75.03230286]]},
    "4": {"name": "4_Karnataka",       "bounds": [[12.80841446, 71.58411407], [14.86688137, 75.03580475]]}, 
    "5": {"name": "5_Kerala",          "bounds": [[7.38235903, 74.23501587], [12.79460812, 77.18965912]]}, 
    "6": {"name": "6_SouthTamilNadu",  "bounds": [[7.01648331, 77.18965912], [10.06777668, 81.33168793]]}, 
    "7": {"name": "7_NorthTamilNadu",  "bounds": [[10.08158302, 79.02596283], [13.51026344, 82.38100433]]}, 
    "8": {"name": "8_SouthAndhraPradesh","bounds": [[13.51026344, 79.87512970], [16.53046989, 83.54584503]]}, 
    "9": {"name": "9_NorthAndhraPradesh","bounds": [[16.53046989, 82.18675232], [19.11697006, 85.88069916]]}, 
    "10": {"name": "10_Odisha",        "bounds": [[19.11697006, 84.77572632], [22.21802711, 89.00102234]]},
    "11": {"name": "11_WestBengal",    "bounds": [[20.71798897, 87.48213196], [22.21802711, 89.00102234]]},
    "12": {"name": "12_lakshadweep",   "bounds": [[8.06546974, 71.29454803], [12.32047653, 74.05455780]]},
    "13": {"name": "13_NorthAndaman",  "bounds": [[10.07467461, 91.52107239], [15.23840141, 94.50333405]]},
    "14": {"name": "14_SouthAndaman",  "bounds": [[6.34685087, 92.12857056], [10.07467461, 94.77947235]]} 
}

# --- GLOBAL DATA LOADING ---
COAST_TRACE = []
try:
    if os.path.exists(COAST_SHAPEFILE_PATH):
        COAST_GDF = gpd.read_file(COAST_SHAPEFILE_PATH)
        x_coords = []
        y_coords = []
        for geom in COAST_GDF.geometry:
            if geom.geom_type == 'LineString':
                x, y = geom.xy
                x_coords.extend(x); y_coords.extend(y)
                x_coords.append(None); y_coords.append(None)
            elif geom.geom_type == 'MultiLineString':
                for part in geom.geoms:
                    x, y = part.xy
                    x_coords.extend(x); y_coords.extend(y)
                    x_coords.append(None); y_coords.append(None)
        
        COAST_TRACE = go.Scatter(
            x=x_coords, y=y_coords, 
            mode='lines', 
            line=dict(color='#333333', width=1), 
            hoverinfo='skip'
        )
except Exception as e:
    print(f"Error loading coastline: {e}")


# --- HELPER FUNCTIONS ---

def get_sector_data_paths(sector_id):
    if str(sector_id) not in SECTOR_BOUNDARIES: return None, None
    folder_name = SECTOR_BOUNDARIES[str(sector_id)]["name"]
    sector_dir = os.path.join(BASE_DATA_DIR, folder_name)
    hist_path = os.path.join(sector_dir, HISTORICAL_ARCHIVE_NC_NAME)
    ckpt_dir = os.path.join(sector_dir, MODEL_CHECKPOINTS_SUBDIR)
    
    if not os.path.exists(hist_path) or not os.path.exists(ckpt_dir):
        return None, None
        
    ckpt_files = glob.glob(os.path.join(ckpt_dir, MODEL_CHECKPOINT_PATTERN))
    if not ckpt_files: return None, None
    
    return hist_path, max(ckpt_files, key=os.path.getmtime).replace('.index', '')

def run_inference_core(tif_bytes, filename, sector_id, timesteps):
    # --- 1. VALIDATION LOGIC ---
    try:
        timesteps = int(timesteps)
    except:
        raise ValueError("Timesteps must be a valid number.")
    
    if timesteps < 1000 or timesteps > 16000:
        raise ValueError("Historical passes must be between 1000 and 16000.")

    hist_path, ckpt_path = get_sector_data_paths(sector_id)
    if not hist_path: raise ValueError("Model files not found for this sector. Please check 'model_data' folder.")
    
    try:
        base = filename
        d_str, t_str = base.split('_')[1], base.split('_')[2]
        ts = pd.to_datetime(d_str + t_str, format='%d%b%Y%H%M')
    except:
        ts = pd.Timestamp.now()

    # --- 2. Load & Check Data Availability ---
    arc = xr.open_dataset(hist_path).rename({'lat': 'y', 'lon': 'x'})
    arc = arc.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326")
    grid_tmpl = arc[VAR_NAME].isel(time=0, drop=True)
    
    # Filter data BEFORE the new timestamp
    data_pre = arc[VAR_NAME].sel(time=slice(None, ts - pd.Timedelta(seconds=1)))
    
    # Check exact number of available steps
    available_steps = data_pre.sizes['time']
    if available_steps < timesteps:
        raise ValueError(f"Insufficient historical data in latest.nc. You requested {timesteps} passes, but only {available_steps} are available before the uploaded date.")

    # Slice
    hist_slice = data_pre.isel(time=slice(-int(timesteps), None))
    
    # --- 3. Process New TIF ---
    new_rio = rioxarray.open_rasterio(io.BytesIO(tif_bytes), masked=True).squeeze()
    new_rio = new_rio.rio.set_crs("EPSG:4326")
    
    bounds = SECTOR_BOUNDARIES[str(sector_id)]["bounds"]
    miny, minx = bounds[0]
    maxy, maxx = bounds[1]
    cropped = new_rio.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs="EPSG:4326")
    resampled = cropped.rio.reproject_match(grid_tmpl, resampling=Resampling.bilinear)
    
    new_np = resampled.values.astype('float32')
    new_np[new_np == MISSING_VALUE] = np.nan
    new_np -= 273.15
    
    orig_da = xr.DataArray(
        data=new_np, coords={'y': grid_tmpl.coords['y'], 'x': grid_tmpl.coords['x']}, dims=['y', 'x']
    ).expand_dims(time=[ts]).rename({'y': 'lat', 'x': 'lon'})
    
    inf_data = xr.concat([hist_slice, orig_da.rename({'lat': 'y', 'lon': 'x'})], dim='time')
    
    uid = str(uuid.uuid4())[:8]
    temp_in = os.path.join(TEMP_DIR, f"in_{uid}.nc")
    temp_out = os.path.join(TEMP_DIR, f"out_{uid}.nc")
    
    ds_in = xr.Dataset({VAR_NAME: inf_data, 'mask': arc['mask']}).rename({'y': 'lat', 'x': 'lon'})
    ds_in[VAR_NAME].encoding['_FillValue'] = -9999.0
    ds_in.to_netcdf(temp_in)
    
    # --- 4. Run DINCAE ---
    lon, lat, time, data, missing, mask = DINCAE.load_gridded_nc(temp_in, VAR_NAME)
    inf_gen, nvar, inf_len, mean_d = DINCAE.data_generator(lon, lat, time, data, missing, train=False, ntime_win=3)
    
    DINCAE.reconstruct_from_checkpoint(
        lon=lon, lat=lat, time=time, mask=mask, meandata=mean_d,
        inference_datagen=inf_gen, inference_len=inf_len,
        outdir=TEMP_DIR, output_filename=f"out_{uid}.nc",
        load_model_path=ckpt_path, nvar=nvar,
        enc_nfilter_internal=[16, 32, 64, 128], skipconnections=[1, 2, 3, 4],
        frac_dense_layer=[0.2], ntime_win=3
    )
    
    with xr.open_dataset(temp_out) as filled:
        rec_slice = filled['mean_rec'].isel(time=[-1]).load()
    
    rec_slice = rec_slice.assign_coords(orig_da.coords)
    hybrid = xr.where(orig_da.isnull(), rec_slice, orig_da)
    
    # --- 5. Plotting (Smoothed) ---
    orig_2d = orig_da.squeeze()
    hybrid_2d = hybrid.squeeze()
    
    vmin = np.nanpercentile(orig_2d.values, 5)
    vmax = np.nanpercentile(orig_2d.values, 95)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    
    im1 = orig_2d.plot.imshow(ax=ax1, cmap='jet', vmin=vmin, vmax=vmax, interpolation='bilinear', add_colorbar=False)
    ax1.set_title("Original (with Gaps)")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    ax1.grid(True, linestyle='--', alpha=0.3)

    im2 = hybrid_2d.plot.imshow(ax=ax2, cmap='jet', vmin=vmin, vmax=vmax, interpolation='bilinear', add_colorbar=False)
    ax2.set_title("Reconstructed (Gap-Filled)")
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("")
    ax2.grid(True, linestyle='--', alpha=0.3)

    fig.colorbar(im2, ax=[ax1, ax2], label='SST (°C)', shrink=0.9, pad=0.02)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    hybrid_ds = hybrid.to_dataset(name=VAR_NAME)
    dl_path = os.path.join(TEMP_DIR, f"final_{uid}.nc")
    hybrid_ds.to_netcdf(dl_path)
    
    if os.path.exists(temp_in): os.remove(temp_in)
    if os.path.exists(temp_out): os.remove(temp_out)
    
    return img_base64, dl_path, ts.strftime('%d-%b-%Y %H:%M')

# --- DASH UI SETUP ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .js-plotly-plot .plotly .cursor-crosshair { cursor: pointer !important; }
            .js-plotly-plot .plotly .cursor-pointer { cursor: pointer !important; }
            .js-plotly-plot .plotly .nsewdrag { cursor: pointer !important; }
            .rounded-box {
                border: 2px solid #005c97 !important;
                border-radius: 15px !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
                overflow: hidden;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

header = html.Div(
    [html.H1("DINCAE Ocean Reconstruction Tool", className="text-white text-center fw-bold", style={'padding': '20px'})],
    style={"background": "linear-gradient(90deg, #005c97 0%, #363795 100%)", "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.3)", "marginBottom": "20px"}
)

def build_map_figure(selected_id=None):
    fig = go.Figure(data=[COAST_TRACE] if COAST_TRACE else [])
    
    for sid, data in SECTOR_BOUNDARIES.items():
        bounds = data['bounds']
        min_lat, min_lon = bounds[0]
        max_lat, max_lon = bounds[1]
        
        x_pts = [min_lon, max_lon, max_lon, min_lon, min_lon]
        y_pts = [min_lat, min_lat, max_lat, max_lat, min_lat]
        
        is_active = str(sid) == str(selected_id)
        fill_color = "rgba(0, 0, 255, 0.3)" if is_active else "rgba(255, 0, 0, 0.05)"
        line_color = "blue" if is_active else "red"
        width = 3 if is_active else 1.5
        
        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts,
            fill="toself", fillcolor=fill_color,
            mode='lines', line=dict(color=line_color, width=width),
            name=data['name'], text=f"Sector {sid}",
            hoverinfo="text+name", customdata=[sid] 
        ))

    fig.update_layout(
        title="Select Ocean Sector",
        dragmode=False,
        margin={"r":0,"t":40,"l":0,"b":0},
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(range=[65, 100], showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[5, 25], showgrid=False, zeroline=False, fixedrange=True),
        height=500
    )
    return fig

app.layout = html.Div([
    header,
    dcc.Store(id='selected-sector-store'),
    dcc.Store(id='inference-result-store'),
    dcc.Download(id="download-nc"),
    
    dbc.Container([
        dbc.Row([
            # COL 1: MAP
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(" Select Sector (Click Map)", className="bg-light fw-bold"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='sector-map', 
                            figure=build_map_figure(),
                            config={'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False},
                            style={'height': '500px', 'width': '100%'}
                        ),
                        html.Div(id='selection-output', className="text-center mt-2 fw-bold text-primary")
                    ])
                ], className="h-100 rounded-box")
            ], width=4),

            # COL 2: CONTROLS
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Configuration", className="bg-light fw-bold"),
                    dbc.CardBody([
                        html.Label("Upload TIF (Indian Ocean)", className="fw-bold"),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'},
                        ),
                        html.Div(id='upload-status', className="mb-3 text-muted small"),
                        
                        html.Label("Historical Timesteps (Passes)", className="fw-bold mt-2"),
                        # Manual Input + Dropdown List
                        dbc.Input(
                            id='timesteps-input', 
                            type='number', 
                            value=3000, 
                            min=1000, 
                            max=16000, 
                            step=1, 
                            list='timesteps-list'
                        ),
                        html.Datalist(
                            id='timesteps-list',
                            children=[html.Option(value=str(i)) for i in range(1000, 16001, 1000)]
                        ),
                        dbc.FormText("Select a preset or type a specific value (1000-16000)"),
                        
                        html.Hr(),
                        dbc.Button("▶ Run Reconstruction", id='run-btn', color="primary", className="w-100", size="lg", disabled=True),
                        dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id="loading-output"))
                    ])
                ], className="h-100 rounded-box")
            ], width=3),

            # COL 3: RESULTS
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Results & Download", className="bg-light fw-bold"),
                    dbc.CardBody([
                        # --- Error Alert Box ---
                        dbc.Alert(id="error-alert", is_open=False, color="danger", dismissable=True),
                        
                        html.Div(id='result-container', style={'display': 'none'}, children=[
                            html.H5(id='res-title', className="text-center text-primary"),
                            html.P(id='res-date', className="text-center text-muted"),
                            html.Img(id='img-combined', style={'width': '100%', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                            html.Hr(),
                            dbc.Button("⬇ Download NetCDF", id="btn-download", color="success", className="w-100")
                        ]),
                        html.Div(id='placeholder-res', children=[html.Div("No results yet.", className="text-center text-muted mt-5")])
                    ])
                ], className="h-100 rounded-box")
            ], width=5),
        ])
    ], fluid=True, style={'paddingBottom': '50px'})
])

@callback(
    [Output('selected-sector-store', 'data'), Output('sector-map', 'figure'), Output('selection-output', 'children')],
    [Input('sector-map', 'clickData')],
    [State('selected-sector-store', 'data')]
)
def update_map_selection(clickData, current_id):
    new_id = current_id
    if clickData:
        try:
            point = clickData['points'][0]
            if 'customdata' in point:
                raw_id = point['customdata']
                new_id = raw_id[0] if isinstance(raw_id, list) else raw_id
        except Exception as e:
            print(f"Click error: {e}")

    fig = build_map_figure(new_id)
    fig.update_layout(uirevision='constant') 
    text = f"Selected: Sector {new_id} ({SECTOR_BOUNDARIES[str(new_id)]['name']})" if new_id else "No Sector Selected"
    return new_id, fig, text

@callback(
    [Output('run-btn', 'disabled'), Output('upload-status', 'children')],
    [Input('upload-data', 'contents'), Input('selected-sector-store', 'data')],
    [State('upload-data', 'filename')]
)
def toggle_button(contents, sector_id, filename):
    status = f"File: {filename}" if filename else "No file uploaded"
    if contents and sector_id: return False, status
    return True, status

@callback(
    [Output('result-container', 'style'), 
     Output('placeholder-res', 'style'),
     Output('img-combined', 'src'), 
     Output('inference-result-store', 'data'), 
     Output('res-title', 'children'),
     Output('res-date', 'children'), 
     Output('loading-output', 'children'),
     Output('error-alert', 'children'),
     Output('error-alert', 'is_open')],
    [Input('run-btn', 'n_clicks')],
    [State('upload-data', 'contents'), State('upload-data', 'filename'),
     State('selected-sector-store', 'data'), State('timesteps-input', 'value')],
    prevent_initial_call=True
)
def run_process(n, contents, filename, sector_id, timesteps):
    if not contents: raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Run Core Logic
        img_combined, dl_path, date_str = run_inference_core(decoded, filename, sector_id, timesteps)
        
        # Success: Return Results, Hide Error
        return (
            {'display': 'block'}, {'display': 'none'}, 
            f"data:image/png;base64,{img_combined}", 
            dl_path, f"Sector {sector_id}: {SECTOR_BOUNDARIES[str(sector_id)]['name']}",
            f"Date: {date_str}", "",
            "", False
        )
    except ValueError as ve:
        # Validation Error: Show Alert
        return (
            {'display': 'none'}, {'display': 'block'}, 
            "", "", "", "", "",
            str(ve), True 
        )
    except Exception as e:
        # General Error: Show Alert
        print(f"Process Error: {e}")
        return (
            {'display': 'none'}, {'display': 'block'}, 
            "", "", "", "", "",
            f"An unexpected error occurred: {e}", True
        )

@callback(
    Output("download-nc", "data"),
    Input("btn-download", "n_clicks"),
    State("inference-result-store", "data"),
    prevent_initial_call=True
)
def download_func(n, nc_path):
    if nc_path and os.path.exists(nc_path): return dcc.send_file(nc_path)
    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)