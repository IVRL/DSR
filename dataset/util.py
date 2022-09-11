import os

import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())
BURST_NUM = int(os.getenv('BURST_NUM'))


def check_full_dataset(folder: str) -> int:
    """Check full dataset.
    Check all scenes, each scenes contains 10 altitudes, and each altitude
    contain one X7 zoom and one burst sequence (JPG+DNG) with 7 frames.

    Args:
        folder (str): Path to full dataset.

    Returns:
        int: Number of problems found.
    """

    altitudes = ['10', '20', '30', '40', '50', '70', '80', '100', '120', '140']

    cnt = 0
    for scene_id in os.listdir(folder):

        for altitude in altitudes:

            if not os.path.isdir(os.path.join(folder, scene_id, altitude)):
                print(f'Scene {scene_id}, Altitude: {altitude}: '
                      f'Missing altitude {altitude} folder.')
                cnt += 1
                continue

            if not os.path.isfile(os.path.join(folder, scene_id, altitude,
                                               'tele.JPG')):
                print(f'Scene {scene_id}, Altitude: {altitude}: '
                      f'Missing Tele camera JPG image.')
                cnt += 1
            
            if not os.path.isfile(os.path.join(folder, scene_id, altitude,
                                               'tele.DNG')):
                print(f'Scene {scene_id}, Altitude: {altitude}: '
                      f'Missing Tele camera DNG image.')
                cnt += 1

            for i in range(BURST_NUM):

                if not os.path.isfile(os.path.join(folder, scene_id, altitude,
                                                   f'hasselblad{i}.JPG')):
                    print(f'Scene {scene_id}, Altitude: {altitude}: '
                          f'Missing Hasselblad burst frame {i} JPG image.')
                    cnt += 1

                if not os.path.isfile(os.path.join(folder, scene_id, altitude,
                                                   f'hasselblad{i}.DNG')):
                    print(f'Scene {scene_id}, Altitude: {altitude}: '
                          f'Missing Hasselblad burst frame {i} DNG image.')
                    cnt += 1
    return cnt
