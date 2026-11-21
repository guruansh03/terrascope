"""Real remote-sensing data: GeoTIFF I/O, band mapping, DEM / WorldCover helpers and dataset loaders.

Heavy dependencies (rasterio, requests, huggingface_hub) are imported lazily inside functions.
"""
from .geotiff import GeoTiffError, register_geotiff  # noqa: F401
