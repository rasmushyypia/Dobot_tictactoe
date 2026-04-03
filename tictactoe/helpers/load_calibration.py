"""
Load Dobot calibration JSON and build per‑cell pick/place grids.

Expected JSON keys
    PICK_X, RETURN_X, PICK_O, RETURN_O           – points on the feeder slide
    BL_CORNER, BR_CORNER, TL_CORNER, TR_CORNER   – board corners (x, y, z, r)

Returns a dict:
    PICK_X, RETURN_X, PICK_O, RETURN_O           – tuples (x,y,z,r)
    TTT_CELLS_PICK, TTT_CELLS_PLACE              – 3×3 lists of tuples
"""

import json
from pathlib import Path
import numpy as np

def load_calibration(path='calib_points.json', place_offset=8):
    data = json.loads(Path(path).read_text())

    # feeder slide points
    PICK_X   = tuple(data['PICK_X'])
    RETURN_X = tuple(data['RETURN_X'])
    PICK_O   = tuple(data['PICK_O'])
    RETURN_O = tuple(data['RETURN_O'])

    # board corners (arrays for vector math)
    bl = np.array(data['BL_CORNER'][:3], float)
    br = np.array(data['BR_CORNER'][:3], float)
    tl = np.array(data['TL_CORNER'][:3], float)
    tr = np.array(data['TR_CORNER'][:3], float)
    r  = float(data['BL_CORNER'][3])           # keep BL orientation for all cells

    # basis vectors (one‑cell step in X and Y on the board surface)
    v_x = (br - bl) / 2.0        # board is 2 cells wide
    v_y = (tl - bl) / 2.0        # board is 2 cells tall

    # bilinear Z interpolation helpers
    def interp_z(alpha, beta):
        """Return Z at fractional (alpha,beta) ∈ [0,1]²."""
        z_bl = bl[2]
        z_br = br[2]
        z_tl = tl[2]
        z_tr = tr[2]
        return ((1 - alpha) * (1 - beta) * z_bl +
                alpha       * (1 - beta) * z_br +
                (1 - alpha) * beta       * z_tl +
                alpha       * beta       * z_tr)

    # build 3×3 grids
    pick_grid  = []
    place_grid = []
    for row in range(3):
        pick_row  = []
        place_row = []
        for col in range(3):
            alpha = col / 2.0    # 0, 0.5, 1
            beta  = row / 2.0
            xyz = bl + col * v_x + row * v_y
            z_pick  = interp_z(alpha, beta)
            z_place = z_pick + place_offset
            pick_row.append((float(xyz[0]), float(xyz[1]), float(z_pick),  r))
            place_row.append((float(xyz[0]), float(xyz[1]), float(z_place), r))
        pick_grid.append(pick_row)
        place_grid.append(place_row)

    return {
        'PICK_X': PICK_X, 'RETURN_X': RETURN_X,
        'PICK_O': PICK_O, 'RETURN_O': RETURN_O,
        'TTT_CELLS_PICK':  pick_grid,
        'TTT_CELLS_PLACE': place_grid,
    }
