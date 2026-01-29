from skimage import measure
import numpy as np
from shapely.geometry import Polygon



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
    simplified = polygon.simplify(tolerance, preserve_topology=True)

    # If simplify returns a MultiPolygon or invalid geometry, fallback
    if simplified.is_empty or not simplified.is_valid:
        return []

    # Extract exterior coords
    simplified_coords = list(simplified.exterior.coords)
    return [[x, y] for x, y in simplified_coords]