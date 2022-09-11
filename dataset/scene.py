import os
import math
import json
from fractions import Fraction
from typing import List

import dotenv
import numpy as np
import cv2 as cv
import rawpy


dotenv.load_dotenv(dotenv.find_dotenv())
BURST_NUM = int(os.getenv('BURST_NUM'))
FLANN_INDEX_KDTREE = int(os.getenv('FLANN_INDEX_KDTREE'))
MIN_MATCH_COUNT = int(os.getenv('MIN_MATCH_COUNT'))
CAMERA_BLACK_LEVEL = int(os.getenv('CAMERA_BLACK_LEVEL'))


class Scene():
    """Scene class.
    One scene contains 1 X7 optical zoom JPG and DNG images, and 7 X1 burst frame
    with both JPG and DNG format.
    """

    folder: str

    hr_jpg_image: np.ndarray
    hr_raw_image: np.ndarray
    lr_jpg_images: List[np.ndarray]
    lr_raw_images: List[np.ndarray]
    raw_metadata: dict

    hr_cropped_jpg: np.ndarray
    hr_cropped_raw: np.ndarray
    lr_cropped_jpg: List[np.ndarray]
    lr_cropped_raw: List[np.ndarray]

    crop_border: int = 300
    
    def __init__(self, folder: str) -> None:
        """Load X7 and X1 JPG images and X1 DNG images, and metadata of DNG images.

        Args:
            folder (str): Path to the folder contains all images in one scene.
        """
        self.folder = folder

        self.hr_jpg_image = cv.imread(os.path.join(folder, 'tele.JPG'))
        self.lr_jpg_images = [cv.imread(os.path.join(folder, f'hasselblad{i}.JPG'))
                              for i in range(BURST_NUM)]

        self.hr_raw_image = self._load_raw_image(
            os.path.join(folder, 'tele.DNG'))
        self.lr_raw_images = [self._load_raw_image(os.path.join(folder, f'hasselblad{i}.DNG'))
                              for i in range(BURST_NUM)]

        self._load_raw_metadata()

    def _load_raw_metadata(self) -> None:
        """Get RAW image metadata."""
        # Tele camera RAW
        with rawpy.imread(os.path.join(self.folder, 'tele.DNG')) as raw:
            self.raw_metadata = {'black_level_per_channel': [raw.black_level_per_channel],
                                 'camera_whitebalance': [raw.camera_whitebalance],
                                 'color_matrix': [raw.color_matrix.tolist()],
                                 'daylight_whitebalance': [raw.daylight_whitebalance],
                                 'white_level': [raw.white_level],
                                 'color_desc': [raw.color_desc.decode()],
                                 }

        # Hasselblad camera RAW
        lr_raw_pathes = [os.path.join(self.folder, f'hasselblad{i}.DNG')
                         for i in range(BURST_NUM)]
        for raw_path in lr_raw_pathes:
            with rawpy.imread(raw_path) as raw:
                self.raw_metadata['black_level_per_channel'].append(
                    raw.black_level_per_channel)
                self.raw_metadata['camera_whitebalance'].append(
                    raw.camera_whitebalance)
                self.raw_metadata['color_matrix'].append(
                    raw.color_matrix.tolist())
                self.raw_metadata['daylight_whitebalance'].append(
                    raw.daylight_whitebalance)
                self.raw_metadata['white_level'].append(
                    raw.white_level)
                self.raw_metadata['color_desc'].append(
                    raw.color_desc.decode())

    @staticmethod
    def _convert_to_four_channels(raw: rawpy._rawpy.RawPy) -> np.ndarray:
        """Convert RAW image into four channel (RGBG).

        Args:
            raw (rawpy._rawpy.RawPy): Load raw data.

        Returns:
            np.ndarray: Four channel raw image with shape (H, W, 4).
        """
        assert raw.color_desc.decode() == 'RGBG'

        image = raw.raw_image_visible.clip(CAMERA_BLACK_LEVEL)
        image = image.astype(np.uint16)

        output = np.zeros(shape=(image.shape[0] // 2, image.shape[1] // 2, 4),
                          dtype=np.uint16)
        output[:, :, 0] = image[np.where(raw.raw_pattern == 0)[0][0]::2,
                                np.where(raw.raw_pattern == 0)[1][0]::2]
        output[:, :, 1] = image[np.where(raw.raw_pattern == 1)[0][0]::2,
                                np.where(raw.raw_pattern == 1)[1][0]::2]
        output[:, :, 2] = image[np.where(raw.raw_pattern == 2)[0][0]::2,
                                np.where(raw.raw_pattern == 2)[1][0]::2]
        output[:, :, 3] = image[np.where(raw.raw_pattern == 3)[0][0]::2,
                                np.where(raw.raw_pattern == 3)[1][0]::2]
        return output

    @classmethod
    def _load_raw_image(cls, file) -> np.ndarray:
        """Load raw iamges and convert to four channel RGBG."""
        with rawpy.imread(file) as raw:
            return cls._convert_to_four_channels(raw)

    def fov_match(self):
        """Crop out matching field of view from each image in the LR burst sequence
        and corresponding raw image. This is doen by estimationg a homography
        between the first image in the LR burst and HR using SIFT and RANSAC.

        Crop out matching field of view in LR JPG and RAW burst frames.

        Returns:
            bool: If find matching filed of view, return True.
        """

        hr_height, hr_width = self.hr_jpg_image.shape[:2]
        lr_height, lr_width = self.lr_jpg_images[0].shape[:2]
        raw_height, raw_width = self.lr_raw_images[0].shape[:2]

        # convert color image to grayscale
        target = cv.cvtColor(self.hr_jpg_image, cv.COLOR_BGR2GRAY)
        source = cv.cvtColor(self.lr_jpg_images[0], cv.COLOR_BGR2GRAY)

        # initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp_tar, des_tar = sift.detectAndCompute(target, None)
        kp_src, des_src = sift.detectAndCompute(source, None)

        # find matched key points
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_tar, des_src, k=2)

        # store all the good matches as per Lowe's ratio test.
        accept = [m for m, n in matches if m.distance < 0.7*n.distance]

        if len(accept) > MIN_MATCH_COUNT:

            dst_pts = np.float32([kp_tar[m.queryIdx].pt
                                  for m in accept]).reshape(-1, 1, 2)
            src_pts = np.float32([kp_src[m.trainIdx].pt
                                  for m in accept]).reshape(-1, 1, 2)
            transform, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

            # Fov match for JPG image.
            lr_coners = np.float32([[0, 0],
                                    [0, hr_height-1],
                                    [hr_width-1, hr_height-1],
                                    [hr_width-1, 0]
                                    ]).reshape(-1, 1, 2)
            lr_coners = cv.perspectiveTransform(lr_coners, transform)

            new_lr_width = round(
                math.hypot(lr_coners[0, 0, 0] - lr_coners[3, 0, 0],
                           lr_coners[0, 0, 1] - lr_coners[3, 0, 1])
            )
            new_lr_height = round(
                math.hypot(lr_coners[0, 0, 0] - lr_coners[1, 0, 0],
                           lr_coners[0, 0, 1] - lr_coners[1, 0, 1])
            )
            lr_x_start = int(round(lr_coners[0, 0, 0]))
            lr_y_start = int(round(lr_coners[0, 0, 1]))
            lr_transform = cv.getPerspectiveTransform(
                lr_coners.reshape(4, 2),
                np.float32(
                    [[lr_x_start, lr_y_start],
                     [lr_x_start, lr_y_start+new_lr_height-1],
                     [lr_x_start+new_lr_width-1, lr_y_start+new_lr_height-1],
                     [lr_x_start+new_lr_width-1, lr_y_start]]
                )
            )

            lr_images = [cv.warpPerspective(image, lr_transform,
                                            (lr_width, lr_height),
                                            cv.INTER_LINEAR,
                                            borderMode=cv.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
                         for image in self.lr_jpg_images]
            self.lr_jpg_images = [image[lr_y_start:lr_y_start+new_lr_height,
                                        lr_x_start:lr_x_start+new_lr_width,
                                        ...]
                                  for image in lr_images]

            # Fov match for RAW data
            new_raw_width = round(
                math.hypot(lr_coners[0, 0, 0] - lr_coners[3, 0, 0],
                           lr_coners[0, 0, 1] - lr_coners[3, 0, 1]) / 2)
            new_raw_height = round(
                math.hypot(lr_coners[0, 0, 0] - lr_coners[1, 0, 0],
                           lr_coners[0, 0, 1] - lr_coners[1, 0, 1]) / 2)
            raw_x_start = int(round(lr_coners[0, 0, 0] / 2))
            raw_y_start = int(round(lr_coners[0, 0, 1] / 2))
            raw_transform = cv.getPerspectiveTransform(
                lr_coners.reshape(4, 2) / 2,
                np.float32(
                    [[raw_x_start, raw_y_start],
                        [raw_x_start, raw_y_start+new_raw_height-1],
                        [raw_x_start+new_raw_width-1, raw_y_start+new_raw_height-1],
                        [raw_x_start+new_raw_width-1, raw_y_start]]
                )
            )
            raw_images = [cv.warpPerspective(image, raw_transform,
                                             (raw_width, raw_height),
                                             cv.INTER_LINEAR,
                                             borderMode=cv.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0, 0))
                          for image in self.lr_raw_images]
            self.lr_raw_images = [image[raw_y_start:raw_y_start+new_raw_height,
                                        raw_x_start:raw_x_start+new_raw_width,
                                        ...]
                                  for image in raw_images]

            return True
        else:
            return False

    def fov_resize(self, width: int = 720, height: int = 540,
                   interpolation=cv.INTER_NEAREST) -> None:
        """Resize all matching field of view (FOV) in LR JPG images to a fixed size.
        For corresponding RAW images, the size is halved.
        Args:
            width (int, optional): Width of resized FOV in JPG images.
                Defaults to 720.
            height (int, optional): Height of resized FOV in JPG images.
                Defaults to 540.
            interpolation (int, optional): Resize interpolation method.
                Defaults to cv.INTER_NEAREST.
        """
        self.lr_jpg_images = [cv.resize(image, (width, height),
                                        interpolation=interpolation)
                              for image in self.lr_jpg_images]
        self.lr_raw_images = [cv.resize(image, (width//2, height//2),
                                        interpolation=interpolation)
                              for image in self.lr_raw_images]

    @staticmethod
    def get_ccorr_normed(image: np.ndarray,
                         target: np.ndarray,
                         interpolation=cv.INTER_CUBIC
                         ) -> float:
        """Get the normalized cross correlation between two images.

        Args:
            image (np.ndarray): Source image.
            target (np.ndarray): Target image, if target image is not the
                same size as the source image, resize it.
            interpolation (int, optional): Resize interpolation method.
                Defaults to cv.INTER_CUBIC.

        Returns:
            float: _description_
        """
        return cv.matchTemplate(
            image,
            cv.resize(target,
                      (image.shape[1], image.shape[0]),
                      interpolation=interpolation),
            cv.TM_CCORR_NORMED
        )[0][0]

    @staticmethod
    def crop_image(image: np.ndarray,
                   y_start: Fraction,
                   y_end: Fraction,
                   x_start: Fraction,
                   x_end: Fraction
                   ) -> np.ndarray:
        assert all(map(lambda x: x.denominator == 1,
                   (y_start, y_end, x_start, x_end)))
        return image[y_start.numerator: y_end.numerator,
                     x_start.numerator: x_end.numerator,
                     ...]

    def crop(self,
             size: int = 180,
             stride: int = 180,
             border: int = 90,
             threshold: float = 0.9,
             ) -> None:
        """Crop matching FOV and HR images in a sliding window manner.
        Filter out incorrect alignment crop using normalized cross
        correlation.

        Args:
            size (int, optional): Crop size. The cropped patches are square.
                Defaults to 180.
            stride (int, optional): Crop stride. Defaults to 90.
            threshold (float, optional): Normalized cross correlation threshold
                for correct alignment. Defaults to 0.9.
        """

        ratio = Fraction(self.hr_jpg_image.shape[0], self.lr_jpg_images[0].shape[0])
        self.ratio = ratio
        self.lr_crop_size = size
        hr_crop_size = size * ratio
        assert hr_crop_size.denominator == 1
        self.hr_crop_size = hr_crop_size.numerator
        
        self.hr_cropped_jpg = []
        self.hr_cropped_raw = []
        self.lr_cropped_jpg = []
        self.lr_cropped_raw = []
        for top in range(border, self.lr_jpg_images[0].shape[0]-size+1, stride):
            for left in range(border, self.lr_jpg_images[0].shape[1]-size+1, stride):

                top = Fraction(top)
                left = Fraction(left)
                cropped_lr_jpg = [self.crop_image(image,
                                                  top,
                                                  top+size,
                                                  left,
                                                  left+size
                                                  )
                                  for image in self.lr_jpg_images]

                cropped_lr_raw = [self.crop_image(image,
                                                  top/2,
                                                  (top+size)/2,
                                                  left/2,
                                                  (left+size)/2
                                                  )
                                  for image in self.lr_raw_images]
                
                cropped_hr_jpg = self.crop_image(self.hr_jpg_image,
                                                 max(Fraction(0), ratio*top-self.crop_border),
                                                 min(Fraction(int(self.hr_jpg_image.shape[0])), ratio*(top+size)+self.crop_border),
                                                 max(Fraction(0), ratio*left-self.crop_border),
                                                 min(Fraction(int(self.hr_jpg_image.shape[1])), ratio*(left+size)+self.crop_border)
                                                 )

                cropped_hr_raw = self.crop_image(self.hr_raw_image,
                                                 max(Fraction(0), ratio*top-self.crop_border)/2,
                                                 min(Fraction(int(self.hr_jpg_image.shape[0])), ratio*(top+size)+self.crop_border)/2,
                                                 max(Fraction(0), ratio*left-self.crop_border)/2,
                                                 min(Fraction(int(self.hr_jpg_image.shape[1])), ratio*(left+size)+self.crop_border)/2
                                                 )
                
                cropped_hr_jpg, cropped_hr_raw = self._local_alignment(cropped_hr_jpg,
                                                                       cropped_hr_raw,
                                                                       cropped_lr_jpg[0]
                                                                       )
                
                if cropped_hr_jpg is not None \
                    and self.get_ccorr_normed(cropped_lr_jpg[0], cropped_hr_jpg) >= threshold:
                        self.hr_cropped_jpg.append(cropped_hr_jpg)
                        self.hr_cropped_raw.append(cropped_hr_raw)
                        self.lr_cropped_jpg.append(cropped_lr_jpg)
                        self.lr_cropped_raw.append(cropped_lr_raw)
    
    def _local_alignment(self, cropped_hr_jpg, cropped_hr_raw, cropped_lr_jpg):
        
        hr_height, hr_width = self.hr_crop_size, self.hr_crop_size
        lr_height, lr_width = self.lr_crop_size, self.lr_crop_size

        # convert color image to grayscale
        target = cv.cvtColor(cropped_lr_jpg, cv.COLOR_BGR2GRAY)
        source = cv.cvtColor(cropped_hr_jpg, cv.COLOR_BGR2GRAY)

        # initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp_tar, des_tar = sift.detectAndCompute(target, None)
        kp_src, des_src = sift.detectAndCompute(source, None)

        # find matched key points
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_tar, des_src, k=2)

        # store all the good matches as per Lowe's ratio test.
        accept = [m for m, n in matches if m.distance < 0.7*n.distance]

        if len(accept) > MIN_MATCH_COUNT:

            dst_pts = np.float32([kp_tar[m.queryIdx].pt
                                  for m in accept]).reshape(-1, 1, 2)
            src_pts = np.float32([kp_src[m.trainIdx].pt
                                  for m in accept]).reshape(-1, 1, 2)
            transform, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

            # Fov match for JPG image.
            hr_coners = np.float32([[0, 0],
                                    [0, lr_height-1],
                                    [lr_width-1, lr_height-1],
                                    [lr_width-1, 0]
                                    ]).reshape(-1, 1, 2)
            hr_coners = cv.perspectiveTransform(hr_coners, transform)

            hr_transform = cv.getPerspectiveTransform(
                hr_coners.reshape(4, 2),
                np.float32(
                    [[0, 0],
                     [0, hr_height-1],
                     [hr_width-1, hr_height-1],
                     [hr_width-1, 0]]
                )
            )

            hr_jpg = cv.warpPerspective(cropped_hr_jpg, hr_transform,
                                        (hr_width, hr_height),
                                        cv.INTER_LINEAR,
                                        borderMode=cv.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
            hr_raw = cv.warpPerspective(cropped_hr_raw, hr_transform,
                                        (hr_width//2, hr_height//2),
                                        cv.INTER_LINEAR,
                                        borderMode=cv.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0, 0))
            return hr_jpg, hr_raw
    
        else:
            return None, None
                         
    def save(self, folder: str) -> None:
        """Save cropped patches.

        Args:
            folder (str): Folder to save cropped patches.
        """
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, 'metadata.json'), 'w') as file:
            json.dump(self.raw_metadata, file)

        for i, (hr_jpg, hr_raw, lr_jpg, lr_raw) in enumerate(zip(self.hr_cropped_jpg,
                                                                 self.hr_cropped_raw,
                                                                 self.lr_cropped_jpg,
                                                                 self.lr_cropped_raw,
                                                                 )):

            os.makedirs(os.path.join(folder, f'{i:02}'), exist_ok=True)

            cv.imwrite(os.path.join(folder, f'{i:02}', 'tele.png'), hr_jpg)
            cv.imwrite(os.path.join(folder, f'{i:02}', 'tele_raw.png'), hr_raw)

            for j, (jpg, raw) in enumerate(zip(lr_jpg, lr_raw)):
                cv.imwrite(os.path.join(folder,
                                        f'{i:02}',
                                        f'hasselblad{j}.png'),
                           jpg)
                cv.imwrite(os.path.join(folder,
                                        f'{i:02}',
                                        f'hasselblad_raw{j}.png'),
                           raw)
