import os
import re
from typing import Tuple, Optional

import shutil
import rawpy
import exifread
import xmltodict
from PIL.ExifTags import TAGS
import tifffile
from GPSPhoto import gpsphoto


class DroneImageFile():
    """Base Drone iamge class."""
    filepath: str
    filename: str
    latitude: float
    longitude: float
    altitude: int
    format: str
    camera: str
    valid_format: tuple

    def __init__(self, filepath: str) -> None:
        """Get relative altitude, camera setting, GPS information from EXIF.

        Args:
            filepath (str): Path to drone image file, either JPG or NDG
        """
        self.filepath = filepath
        self.filename = os.path.basename(self.filepath)
        self._check_format()
        self._get_altitude()
        self._check_camera_settings()
        self._get_gps_info()
        
    def _check_format(self) -> None:
        """Check image format"""
        self.format = os.path.splitext(self.filename)[-1]
        assert self.format in self.valid_format, \
            f'{self.filename}: invalid image format.'
            
    def _get_altitude(self) -> None:
        raise NotImplementedError

    def _get_gps_info(self) -> None:
        """Get GPS information (latitude and longitude)"""
        gpsdata = gpsphoto.getGPSData(self.filepath)
        self.latitude = round(gpsdata["Latitude"], 4)
        self.longitude = round(gpsdata["Longitude"], 4)

    def _check_camera_settings(self) -> None:
        """Check camera settings and get optical zoom ratio."""
        with open(self.filepath, 'rb') as file:
            exifdata = exifread.process_file(file)
            metadata = {TAGS.get(tag, tag): value
                        for tag, value in exifdata.items()}
            width = int(str(metadata['Image ImageWidth']))
            height = int(str(metadata['Image ImageLength']))
            zoom_ratio = int(str(metadata['EXIF DigitalZoomRatio']))
            self.camera =  str(metadata['Image Make'])

        # check digital zoom is disabled
        assert zoom_ratio == 1, \
            f"{self.filename}: wrong digital zoom."

        # check image size
        assert (height, width) in [(3000, 4000),    # Tele camera JPG
                                   (3956, 5280),    # Hasselblad camera JPG
                                   (120, 160),      # Hasselblad and Tele camera DNG
                                   ], \
            f"{self.filename}: wrong size."

    def _folder_to_save(self, folder: str, scene_id: int) -> str:
        """Get folder path where the iamge to be saved in full dataset.

        Args:
            folder (str): Path to the full dataset.
            scene_id (int): Scene id of the iamge.

        Returns:
            str: Folder path where the iamge to be saved in full dataset
        """
        folder = os.path.join(folder, f"{scene_id:04}", str(self.altitude))
        os.makedirs(folder, exist_ok=True)
        return folder

    def get_location(self) -> Tuple[float, float]:
        """Get the GPS location of the image.

        Returns:
            Tuple[float, float]: (latitude, longitude)
        """
        return self.latitude, self.longitude

    def save(self,
             folder: str,
             scene_id: int,
             burst_id: Optional[int] = None,
             ) -> None:
        """Rename the image and save to full dataset folder.

        Args:
            folder (str): Path to the full dataset.
            scene_id (int): Image scene id.
            burst_id (int): Image burst id.
        """
        folder = self._folder_to_save(folder, scene_id)

        if self.camera == 'DJI':
            name = f"tele{self.format}"
        elif self.camera == 'Hasselblad':
            name = f"hasselblad{burst_id}{self.format}"
        else:
            raise ValueError(f"{self.filename}: Unknown camera")
        
        shutil.copyfile(self.filepath, os.path.join(folder, name))


class DroneRAW(DroneImageFile):
    """Drone DNG image class.

    Raise:
        rawpy.LibRawFileUnsupportedError: the DNG image is damanged.
    """
    
    valid_format  = ('.DNG', '.dng')
    
    def __init__(self, filepath: str) -> None:
        """Check file with DNG extension,
        Check weather the DNG file is damanged or not.

        Args:
            filepath (str): Path to drone DNG image file.
        """
        self._test_raw(filepath=filepath)
        super().__init__(filepath=filepath)

    @staticmethod
    def _test_raw(filepath) -> None:
        """Check weather the DNG file is damanged or not.

        Raise:
            rawpy.LibRawFileUnsupportedError: the DNG image is damanged.
        """
        with rawpy.imread(filepath) as raw:
            raw.postprocess()
        return

    def _get_altitude(self) -> None:
        """Get relative altitude from EXIF data."""
        tiffexifdict = dict()
        with tifffile.TiffFile(self.filepath) as tif:
            for page in tif.pages:
                for tag in page.tags:
                    tag_name, tag_value = tag.name, tag.value
                    tiffexifdict[tag_name] = tag_value
            metadata = xmltodict.parse(tiffexifdict['XMP'])
            altitude = eval((metadata['x:xmpmeta']
                             ['rdf:RDF']
                             ['rdf:Description']
                             ['@drone-dji:RelativeAltitude']))
        self.altitude = int(round(altitude, -1))


class DroneRGB(DroneImageFile):
    """Drone JPG image class.

    Raise:
        rawpy.LibRawFileUnsupportedError: the corresponding DNG image
            is damanged.
    """
    
    valid_format  = ('.jpg', '.JPG', '.jpeg', '.JPEG')
    
    def __init__(self, filepath: str) -> None:
        """Check file with JPG extension,
        """
        # assert filepath.endswith(".JPG")
        super().__init__(filepath=filepath)

    def _get_altitude(self) -> None:
        pass
             
    def set_altitude(self, altitude: int) -> None:
        self.altitude = altitude
