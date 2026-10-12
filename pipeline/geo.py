"""Shared local coordinate frame: UTM 18N meters relative to an origin in Midtown. x=east, y=north, z=up (NAVD88 m)."""
import numpy as np
from pyproj import Transformer

ORIGIN_LAT, ORIGIN_LON = 40.75, -73.98
_fwd = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
_inv = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
OX, OY = _fwd.transform(ORIGIN_LON, ORIGIN_LAT)
FT = 0.3048

# UTM grid north differs from true north by the meridian convergence; headings must be corrected.
CONVERGENCE_DEG = float(np.degrees(np.arctan(np.tan(np.radians(-75 - ORIGIN_LON) * -1) * np.sin(np.radians(ORIGIN_LAT)))))

def to_local(lon, lat):
    x, y = _fwd.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x) - OX, np.asarray(y) - OY

def to_lonlat(x, y):
    return _inv.transform(np.asarray(x) + OX, np.asarray(y) + OY)

def true_heading_to_grid(h):
    """Compass (true-north) heading in degrees -> heading in the local UTM grid frame."""
    return (h - CONVERGENCE_DEG) % 360
