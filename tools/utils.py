import pandas as pd
from sklearn.metrics import confusion_matrix

import numpy as np
from skimage import measure
# from shapely.geometry import Polygon, LinearRing
from shapely.geometry import Polygon, MultiPolygon, MultiLineString

def get_exterior_coords(geom):
    if isinstance(geom, Polygon):
        return [list(geom.exterior.coords)]
    elif isinstance(geom, MultiPolygon):
        return [list(p.exterior.coords) for p in geom.geoms]
    elif isinstance(geom, MultiLineString):
        return list(geom.coords)
    else:
        # No polygon exterior available (e.g. MultiLineString)
        return []

def mask_to_polygon(mask: np.ndarray, tolerance: float = 1.0):
    """
    Convert a boolean mask to a simplified polygon (only corners).

    Args:
        mask (np.ndarray): 2D boolean numpy array.
        tolerance (float): Simplification tolerance (higher = more aggressive).

    Returns:
        List[List[float]]: Simplified polygon as [[x0, y0], ..., [x0, y0]].
    """
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded.astype(float), level=0.5)
    if not contours:
        print('no contour found')
        return []

    # Use the longest contour
    contour = max(contours, key=len)
    # Convert (row, col) to (x, y)
    coords = [(float(x), float(y)) for y, x in contour]

    # Ensure it's closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Create shapely polygon and simplify
    polygon = Polygon(coords)
    # if polygon.is_valid is not True:
    #     polygon = make_valid(polygon)
    simplified = polygon.simplify(tolerance, preserve_topology=False)

    # If simplify returns a MultiPolygon or invalid geometry, fallback
    if simplified.is_empty or not simplified.is_valid:
        return []

    # Extract exterior coords
    simplified_coords = list(simplified.exterior.coords)
    # simplified_coords = get_exterior_coords(simplified)
    return [[x, y] for x, y in simplified_coords]


def get_confusion_matrix(y_true, y_pred):

    # # True labels
    # y_true = [0, 1, 0, 2, 1, 2, 0, 1, 2, 2]
    # # Predicted labels
    # y_pred = [0, 0, 0, 2, 1, 2, 1, 1, 2, 2]

    labels = list(set(y_pred + y_true))
    if len(labels) == 1:
        print(labels[0])
        return pd.DataFrame([[len(y_pred)]], index=labels, columns=labels)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # cm = confusion_matrix(y_true, y_pred, labels=y_true)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # print(cm)
    return pd.DataFrame(cm, index=labels, columns=labels)