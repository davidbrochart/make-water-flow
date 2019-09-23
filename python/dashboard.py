import pandas as pd
import numpy as np
import bqplot as bq
from ipywidgets import HBox, Label, Button, ToggleButtons, FileUpload
import ipydatetime
from ipyleaflet import Map, Popup, Marker, GeoJSON, basemaps, ImageOverlay
import json
import os
from datetime import datetime, timedelta
import PIL
from base64 import b64encode
from io import BytesIO
import matplotlib.pyplot as plt

import xarray as xr
from gcsfs import mapping
from dask_kubernetes import KubeCluster as Cluster
from dask.distributed import Client

class Map_menu(object):
    def __init__(self, m, label, file_upload):
        self.m = m
        self.label = label
        self.file_upload = file_upload
        self.coord = None
        self.s = None
        self.p = None
        self.show_menu = False
        self.marker = None
        self.geojson = None
        self.marker_or_geojson = None
        self.current_io = None
    def show(self, **kwargs):
        if not self.show_menu:
            if kwargs.get('type') == 'contextmenu':
                self.show_menu = True
                options = ['Show marker']
                if self.geojson is None:
                    options += ['Show GeoJSON']
                if self.marker_or_geojson is not None:
                    options += [f'Remove {self.marker_or_geojson}']
                options += ['Close']
                self.s = ToggleButtons(options=options, value=None)
                self.s.observe(self.get_choice, names='value')
                self.p = Popup(location=self.coord, child=self.s, max_width=160, close_button=False, auto_close=True, close_on_escape_key=False)
                self.m.add_layer(self.p)
            elif kwargs.get('type') == 'mousemove':
                self.coord = kwargs.get('coordinates')
    def get_choice(self, x):
        self.show_menu = False
        self.s.close()
        self.m.remove_layer(self.p)
        self.p = None
        choice = x['new']
        if choice == 'Show marker':
            self.show_marker()
        elif choice == 'Show GeoJSON':
            self.show_geojson()
        elif choice == 'Remove marker':
            self.remove_marker()
        elif choice == 'Remove GeoJSON':
            self.remove_geojson()
        elif choice == 'Close':
            pass
    def show_geojson(self, *args):
        data = json.loads(list(self.file_upload.value.values())[0]['content'].decode('ascii'))
        self.remove_marker()
        self.remove_geojson()
        self.geojson = GeoJSON(data=data, style = {'color': 'green'})#, 'opacity': 1})#, 'fillOpacity':0.1})
        self.m.add_layer(self.geojson)
        self.marker_or_geojson = 'GeoJSON'
    def show_marker(self):
        self.remove_marker()
        self.remove_geojson()
        self.marker = Marker(location=self.coord)
        self.m.add_layer(self.marker)
        self.marker_or_geojson = 'marker'
    def remove_marker(self):
        if self.marker is not None:
            self.m.remove_layer(self.marker)
            self.marker = None
        self.marker_or_geojson = None
    def remove_geojson(self):
        if self.geojson is not None:
            self.m.remove_layer(self.geojson)
            self.geojson = None
        self.marker_or_geojson = None

label = Label()
file_upload = FileUpload(accept='.geojson,.json',  multiple=False, description='Upload GeoJSON')

m = Map(center=(-10, -60), zoom=4, interpolation='nearest', basemap=basemaps.CartoDB.DarkMatter)
map_menu = Map_menu(m, label, file_upload)
m.on_interaction(map_menu.show)
file_upload.observe(map_menu.show_geojson, 'value')

def get_img(a_web):
    a_norm = a_web - np.nanmin(a_web)
    vmax = np.nanmax(a_norm)
    if vmax != 0:
        a_norm = a_norm / vmax
    a_norm = np.where(np.isfinite(a_web), a_norm, 0)
    a_im = PIL.Image.fromarray(np.uint8(plt.cm.viridis(a_norm)*255))
    a_mask = np.where(np.isfinite(a_web), 255, 0)
    mask = PIL.Image.fromarray(np.uint8(a_mask), mode='L')
    im = PIL.Image.new('RGBA', a_norm.shape[::-1], color=None)
    im.paste(a_im, mask=mask)
    f = BytesIO()
    im.save(f, 'png')
    data = b64encode(f.getvalue())
    data = data.decode('ascii')
    imgurl = 'data:image/png;base64,' + data
    return imgurl

def overlay(m, current_io, da, label):
    imgurl = get_img(da.values)
    bounds = [(
        da.lat.values[-1] - 0.5 * 0.1,
        da.lon.values[0] - 0.5 * 0.1
        ), (
        da.lat.values[0] + 0.5 * 0.1,
        da.lon.values[-1] + 0.5 * 0.1)
        ]
    io = ImageOverlay(url=imgurl, bounds=bounds, opacity=0.5)
    if current_io is not None:
        m.remove_layer(current_io)
    m.add_layer(io)
    return io

def _get_precipitation(ds, map_menu, line, label, at_time=None, from_time=None, to_time=None):
    def _(w):
        if map_menu.marker is None:
            if at_time is not None:
                t = at_time.value
                da = ds.precipitationCal.sel(time=t, method='nearest').compute()
            else:
                t0 = from_time.value
                t1 = to_time.value
                da = ds.precipitationCal.sel(time=slice(t0, t1)).sum(['time']).compute()
                label.value = str(ds.precipitationCal.sel(time=slice(from_time.value, to_time.value)).time)
            io = overlay(map_menu.m, map_menu.current_io, da, label)
            map_menu.current_io = io
        else:
            lat, lon = map_menu.marker.location
            #label.value = str(ds)
            da = ds.precipitationCal.sel(lat=lat, lon=lon, method='nearest').compute()
            s = da.to_series()
            line.x = s.index.values
            line.y = s
    return _

user = os.environ.get('USER')
if user == 'jovyan':
    # we are in Pangeo binder, cluster client
    cluster = Cluster(n_workers=10)
    client = Client(cluster)
else:
    # local client
    client = Client()

zarr = mapping.GCSMap('pangeo-data/gpm_imerg/early_test/chunk_time')
ds_time = xr.open_zarr(zarr)

zarr = mapping.GCSMap('pangeo-data/gpm_imerg/early_test/chunk_space')
ds_space = xr.open_zarr(zarr)

def create_plot():
    index = pd.date_range(start='2000-06-01', end='2001-06-01', freq='30min') + timedelta(minutes=15)
    s = pd.Series(np.full(len(index), np.nan), index=index)

    x = index.values
    y = s

    x_sc = bq.DateScale()
    y_sc = bq.LinearScale(min=0)
    
    line = bq.Lines(x=x, y=y, scales={'x': x_sc, 'y': y_sc}, 
            #display_legend=True, labels=["line 1"],
            #fill='bottom', # opacity does work with this option
            #fill_opacities = [0.5] *len(x)
            )

    panzoom = bq.PanZoom(scales={'x': [x_sc], 'y': [y_sc]})
    #p = bq.Scatter(x=x, y=y, scales= {'x': x_sc, 'y': y_sc})
    ax_x = bq.Axis(scale=x_sc)
    ax_y = bq.Axis(scale=y_sc, orientation='vertical', tick_format='0.2f')
    
    #fig = bq.Figure(marks=[line, p], axes=[ax_x, ax_y])
    fig = bq.Figure(marks=[line], axes=[ax_x, ax_y], interaction=panzoom)
    fig.layout.width = '95%'
    
    #p.interactions = {'click': 'select', 'hover': 'tooltip'}
    #p.selected_style = {'opacity': 1.0, 'fill': 'DarkOrange', 'stroke': 'Red'}
    #p.unselected_style = {'opacity': 0.5}
    #p.tooltip = bq.Tooltip(fields=['x', 'y'], formats=['', '.2f'])
    
    #sel = bq.interacts.IndexSelector(scale=p.scales['x'])
    #
    #def update_range(*args):
    #    if sel.selected:
    #        print(sel.selected[0])
    #    
    #sel.observe(update_range, 'selected')
    #fig.interaction = sel

    return fig, line

fig, line = create_plot()

from_time = ipydatetime.DatetimePicker(value=datetime(2000, 6, 1), min=datetime(2000, 6, 1), max=datetime(2019, 9, 1))
to_time = ipydatetime.DatetimePicker(value=datetime(2000, 8, 31), min=datetime(2000, 6, 1), max=datetime(2019, 9, 1))
go_time_range = Button(description='Go!', tooltip='Show accumulated precipitation')
at_time = ipydatetime.DatetimePicker(value=datetime(2000, 6, 1), min=datetime(2000, 6, 1), max=datetime(2019, 9, 1))
go_at_time = Button(description='Go!', tooltip='Show instant precipitation')

get_precipitation_at_time = _get_precipitation(ds_time, map_menu, line, label, at_time=at_time)
get_precipitation_time_range = _get_precipitation(ds_time, map_menu, line, label, from_time=from_time, to_time=to_time)

go_at_time.on_click(get_precipitation_at_time)
go_time_range.on_click(get_precipitation_time_range)

show_precipitation_time_range = HBox([Label('Show precipitation from'), from_time, Label('to'), to_time, go_time_range])
show_precipitation_at_time = HBox([Label('Show precipitation at'), at_time, go_at_time])
