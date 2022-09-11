import os
import json
import shutil
import argparse
import collections
from datetime import datetime
from tempfile import TemporaryDirectory as TempDir

import dotenv
import rawpy
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import cv2 as cv

from drone import DroneRAW, DroneRGB
from scene import Scene
from util import check_full_dataset
from sync import download_from_drive, upload_to_drive, move_drive

dotenv.load_dotenv(dotenv.find_dotenv())
dotenv.load_dotenv(dotenv.find_dotenv('.env.local'))
BURST_NUM = int(os.getenv('BURST_NUM'))
DRIVE_RAW_ID = os.getenv('DRONESR_X7_RAW_ID')
DRIVE_FULL_ID = os.getenv('DRONESR_X7_FULL_ID')
DRIVE_CROP_ID = os.getenv('DRONESR_X7_CROP_ID')
DRIVE_HISTORY_ID = os.getenv('DRONESR_X7_HISTORY_ID')
LOCAL_CROP_DIR = os.getenv('LOCAL_CROP_DIR')
LOCAL_FULL_DIR = os.getenv('LOCAL_FULL_DIR')


def main(sync_drive: bool = True, sync_crop: bool = False) -> None:
    """Update dataset, append new data to existing dataset.

    Args:
        sync_drive (bool, optional): Synchronize dataset with google drive.
            Defaults to True.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive-raw', default=None, type=str, 
                        help='Google Drive Raw image folder id')
    args = parser.parse_args()

    local_raw_folder = None
    local_full_folder = LOCAL_FULL_DIR
    # Local folder keeps cropped images.
    local_crop_folder = LOCAL_CROP_DIR

    if sync_drive:
        # Get Google drive authentication
        gauth = GoogleAuth()
        gauth.CommandLineAuth()
        drive = GoogleDrive(gauth)
    
    with TempDir(dir=os.getcwd(), prefix="tmp_raw_") as temp_raw_folder, \
            TempDir(dir=os.getcwd(), prefix="tmp_full_") as temp_full_folder, \
            TempDir(dir=os.getcwd(), prefix="tmp_crop_") as temp_crop_folder:

        # Get raw files
        if local_raw_folder:
            temp_raw_folder = local_raw_folder
        else:
            print('Downloding raw data from Google Drive ...')
            download_from_drive(local_folder=temp_raw_folder,
                                drive=drive,
                                drive_id=args.drive_raw if args.drive_raw else DRIVE_RAW_ID,
                                )
            input('Continue?')

        # Sort and rename drone images
        raw_images = list(set(os.path.splitext(file)[0] for file in os.listdir(temp_raw_folder) 
                              if file.endswith(('JPG', 'DNG'))))
        
        # Load location to scene id map
        loc2idx_path = './DroneSR-X7/loc2idx.json'
        loc2idx = {}
        if os.path.isfile(loc2idx_path):
            with open(loc2idx_path, 'r') as file:
                loc2idx = json.load(file)
                loc2idx = {eval(key): int(value) for key, value in loc2idx.items()}

        start_scene_id = max(loc2idx.values()) + 1
        scene_id_record = start_scene_id
        added_scene_ids = set()
        
        bursts = collections.deque(maxlen=BURST_NUM)
        print('Sorting raw images ...')
        for filename in sorted(raw_images):

            imagepath = os.path.join(temp_raw_folder, filename)
            
            if not os.path.isfile(imagepath + '.DNG'):
                print(f'DNG file missed: {filename}')
                bursts.clear()
                continue
            
            if not os.path.isfile(imagepath + '.JPG'):
                print(f'JPG file missed: {filename}')
                bursts.clear()
                continue
            
            try:
                rawfile = DroneRAW(imagepath+'.DNG')
                rgbfile = DroneRGB(imagepath+'.JPG')
                rgbfile.set_altitude(altitude=rawfile.altitude)
            except rawpy.LibRawFileUnsupportedError:
                print(f'DNG file damaged: {filename}')
                bursts.clear()
                continue
                
            # Get scene id
            location = rawfile.get_location()
            if location in loc2idx:
                scene_id = loc2idx[location]
            else:
                scene_id = scene_id_record
                scene_id_record += 1
                loc2idx[location] = scene_id
                added_scene_ids.add(scene_id)

            if rawfile.camera == 'DJI':
                rawfile.save(temp_full_folder, scene_id)
                rgbfile.save(temp_full_folder, scene_id)
                bursts.clear()
            elif rawfile.camera == 'Hasselblad':
                bursts.append((rawfile, rgbfile))
                if len(bursts) == BURST_NUM:
                    for i, (raw, rgb) in enumerate(bursts):
                        raw.save(temp_full_folder, scene_id, i)
                        rgb.save(temp_full_folder, scene_id, i)
            
        # Check all scenes, each scenes contains 10 altitudes, and each
        # altitude contain one X7 zoom and one burst sequence (JPG+DNG)
        # with 7 frames.
        problems = check_full_dataset(temp_full_folder)
        inputs = 'yes'
        while problems and inputs=='yes':
            print("Please manually solve the problems in"
                  f"{temp_full_folder} folder")
            inputs = input('Solved all problems and continue? ([yes]/no/force)? ')
            if inputs == 'no':
                exit()
            elif not inputs == 'force':
                inputs = 'yes'
            problems = check_full_dataset(temp_full_folder)
        print('No problem found :-)' if inputs =='yes' else 'Force update.')

        
        if len(added_scene_ids) > 0:
            # Reset scenes index
            old2new = {}
            deleted_scene_ids = added_scene_ids
            for new_idx, old_idx in enumerate(sorted(os.listdir(temp_full_folder)),
                                            start=start_scene_id):
                deleted_scene_ids.remove(int(old_idx))
                old2new[int(old_idx)] = new_idx
                os.rename(os.path.join(temp_full_folder, old_idx),
                        os.path.join(temp_full_folder, f'{new_idx:04}'))
            remove_locations = []
            for location, scene_id in loc2idx.items():
                if scene_id in deleted_scene_ids:
                    remove_locations.append(location)
            for location in remove_locations:
                loc2idx.pop(location)
            for location, scene_id in loc2idx.items():
                if scene_id in old2new:
                    loc2idx[location] = old2new[scene_id]
            
            # Update loc2idx
            with open(loc2idx_path, 'w') as file:
                loc2idx = {str(key): value for key, value in loc2idx.items()}
                loc2idx = json.dump(loc2idx, file)

        # Add to local full dataset
        if local_full_folder:
            print('Appending to local full dataset ...')
            shutil.copytree(temp_full_folder,
                            local_full_folder,
                            dirs_exist_ok=True)

        # Upload full dataset to drive
        if sync_drive:
            print('Uploading full dataset to drive ...')
            for scene in os.listdir(temp_full_folder):
                upload_to_drive(drive=drive,
                                drive_id=DRIVE_FULL_ID,
                                local=os.path.join(temp_full_folder, scene)
                                )
        if local_full_folder:
            print('Appending to local full dataset ...')
            shutil.copytree(temp_full_folder,
                            local_full_folder,
                            dirs_exist_ok=True)
            
        
        # FOV match and crop
        # Process each scene and each altitude.
        print('Cropping ...')
        for scene_id in os.listdir(temp_full_folder):
            for altitude in os.listdir(os.path.join(temp_full_folder,
                                                    scene_id)):
                scene = Scene(os.path.join(temp_full_folder,
                                           scene_id,
                                           altitude))
                if scene.fov_match():
                    try:
                        scene.fov_resize(width=720, height=540)
                        scene.crop(size=180, stride=180, border=90, threshold=0.9)
                        scene.save(os.path.join(temp_crop_folder,
                                                scene_id,
                                                altitude))
                    except cv.error as e:
                        print(f"Scene {scene_id}, altitude {altitude} error: {e}")
                        continue

        print('Appending to local crop dataset ...')
        shutil.copytree(temp_crop_folder,
                        local_crop_folder,
                        dirs_exist_ok=True)

        if sync_drive and sync_crop:
            print('Uploading crop dataset to drive ...')
            for scene in os.listdir(temp_crop_folder):
                upload_to_drive(drive=drive,
                                drive_id=DRIVE_CROP_ID,
                                local=os.path.join(temp_crop_folder, scene)
                                )

        if not local_raw_folder and sync_drive:
            print('Moving Drive raw folder to history ...')
            date = datetime.now().strftime('%Y-%m-%d-%H-%M')
            history_meta = {'title': date,
                            'parents': [{'id': DRIVE_HISTORY_ID}],
                            'mimeType': 'application/vnd.google-apps.folder'}
            history_folder = drive.CreateFile(history_meta)
            history_folder.Upload()
            move_drive(drive, 
                       args.drive_raw if args.drive_raw else DRIVE_RAW_ID,
                       history_folder['id'])

    print("Done!")


if __name__ == '__main__':
    main()
