import numpy as np
from sentinelhub import CRS, BBox, bbox_to_dimensions, DataSource,\
    SentinelHubRequest, MimeType, WebFeatureService
from datetime import timedelta
import matplotlib.pyplot as plt
from shapely.geometry import Point

m_to_deg = 111111
one_day = timedelta(days=1)

def get_square_around_point(lon, lat, m):
    """
    Given a point p on the earth and a number m, this functions returns the
    coordinates that represent the square area centered at p.
    ...
    Parameters
    ----------
    lon: float
        The longitude of the point (-180 <= lon < 180).
    lat: float
        The latitude of the point (-90 <= lat <= 90).
    m: int
        The size of each side of the square, in meters
    ...
    Returns
    -------
    list:
        A list with 4 float elements representing, in this order, the minimum
        longitude, the minimum latitude, the maximum longitude and the
        maximum latitude. All of them given in decimal degrees.
    """
    half_side = m / 2
    x_min = lon - half_side / (m_to_deg * np.cos(np.deg2rad(lat)))
    x_max = lon + half_side / (m_to_deg * np.cos(np.deg2rad(lat)))
    y_min = lat - half_side / m_to_deg
    y_max = lat + half_side / m_to_deg
    # The coordinates need to be given in the format:
    # [min_lon, min_lat, max_lon, max_lat]
    return [x_min, y_min, x_max, y_max]

def get_bbox_of_given_size(coords, pix, resolution):
    """
    Given a list with coordinates representing a square in the earth's surface,
    this function returns a BBox object from the sentinelhub package with the
    given requirements.
    ...
    Parameters
    ----------
    coords: list
        A list with 4 float elements representing, in this order, the minimum
        longitude, the minimum latitude, the maximum longitude and the
        maximum latitude. All of them given in decimal degrees.
        Coordinates with such format might be obtained via the function
        get_square_around_point.
    pix: tuple
        A 2-dimensional tuple encoding the expected number of pixels of the 
        image to be obtained with the Bbox.
    resolution: int
        Resolution of the image, in meters. Minimum value is 10.
    ...
    Returns
    -------
    sentinelhub.Bbox:
        A Bbox object with the given specifications.
    """
    x_min, y_min, x_max, y_max = coords
    box = BBox(bbox=[x_min, y_min, x_max, y_max], crs=CRS.WGS84)
    box_size = bbox_to_dimensions(box, resolution=resolution)
    if box_size == pix:
        return box
    x_max = (x_max - x_min) * pix[0] / box_size[0] + x_min
    y_max = (y_max - y_min) * pix[1] / box_size[1] + y_min
    box = BBox(bbox=[x_min, y_min, x_max, y_max], crs=CRS.WGS84)
    return box

def get_request_for_saving_picture(box, box_size, config, evalscript, date,\
                                   folder, time='now', interval=5):
    """
    Given all the necessary information required by SentinelHubRequest, as well
    as well as some requirements regarding when the desired picture was taken,
    this function returns a sentinelhub.SentinelHubRequest object that allows
    you to execute a request to save a tiff picture in the specified location.
    """
    if time == 'now': 
        if interval==1:
            time_interval = (str(date).split()[0], str(date+one_day).split()[0])
        elif interval%2 == 0:
            time_interval = (str(date - int(interval/2)*one_day).split()[0],\
                             str(date + int((interval+1)/2)*one_day).split()[0])
        else:
            time_interval = (str(date - int(interval/2)*one_day).split()[0], \
                             str(date + int(interval/2)*one_day).split()[0])
    elif time == 'after':
        time_interval = (str(date + one_day).split()[0], \
                         str(date + interval*one_day).split()[0])
    elif time == 'before':
        time_interval = (str(date - one_day).split()[0], \
                         str(date - interval*one_day).split()[0])
        
    request = SentinelHubRequest(
        data_folder = folder,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_source=DataSource.SENTINEL2_L1C,
                time_interval=time_interval,
                mosaicking_order='leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=box,
        size=box_size,
        config=config
    )
        
    return request

def get_available_dates(bbox, time_interval, config, max_cc=1.0):
    """
    Given a sentinelhub.BBox and a time interval, this function returns a list
    of dates in which an image was taken by the satellite SENTINEL_2_L1C of
    the area defined by the bbox object.
    ...
    Parameters
    ----------
    bbox: sentinelhub.BBox
        An object defining the area you want the picture from.
    time_interval: tuple (str, str)
        The first element defines the starting date and the second one defines
        the end date. Format of the dates is 'YYYY-MM-DDTHH:MM:SS' or
        'YYYY-MM-DD'.
    config: sentinelhub.SHConfig
        IMPORTANT: The instance_id of the configuration needs to be specified.
    ...
    Returns
    -------
    List:
        A list of dates in which a picture of the area defined by the BBox is
        available. The dates are in the format 'YYYY-MM-DD'.
    """
    wfs_iterator = WebFeatureService(
        bbox,
        time_interval,
        data_source=DataSource.SENTINEL2_L1C,
        maxcc=max_cc,
        config=config
    )
    
    available_dates = []
    for tile_info in wfs_iterator:
        available_dates.append(tile_info['properties']['date'])
    return available_dates

def get_field_pixels(img, bbox, field):
    """
    Given the image of a field (rectangular), this functions returns a mask
    indicating which pixels of the image belong to the field.
    ...
    Parameters
    ----------
    img: numpy.ndarray
        Image of the field, in the form of a numpy.ndarray.
        
    bbox: sentinelhub.BBox
        BBox of the image, containing the information about the location on
        earth.
        
    field: shapely.geometry.polygon.Polygon
        Polygon that represents the field on the given CRS, which needs to be
        consistent with the one of the BBox.
    ...
    Returns
    -------
    boolean np.ndarray of shape img.shape
    """
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
    n_x, n_y = mask.shape
    x_len = bbox.max_x - bbox.min_x
    y_len = bbox.max_y - bbox.min_y
    for i,j in np.ndindex(mask.shape):
        x = bbox.min_x + (2*i+1) * x_len / (2 * n_x)
        y = bbox.min_y + (2*j+1) * y_len / (2 * n_y)
        p = Point(x, y)
        mask[i,j] = field.contains(p)
    return mask
        
def plot_image(image, factor=1.0, clip_range = None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    
    