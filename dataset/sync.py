import os
from pydrive.drive import GoogleDrive


def download_from_drive(drive: GoogleDrive,
                        local_folder: str,
                        drive_id: str,
                        ) -> None:
    """Download content from Google drive.

    Args:
        drive (GoogleDrive): Authenticated drive.
        local_folder (str): Path to local folder to keep downloaded file.
        drive_id (str): Google drive file id to download.
    """
    drive_folder = drive.ListFile(
        {'q': f"'{drive_id}' in parents and trashed=false",
         'supportsAllDrives': True,
         'includeItemsFromAllDrives': True
         }
        ).GetList()
        
    for file in drive_folder:
                    
        filepath = os.path.join(local_folder, file['title'])
        if not file['mimeType'].endswith('folder'):
            file.GetContentFile(filepath)
        else:
            os.makedirs(filepath, exist_ok=True)
            download_from_drive(drive, filepath, file['id'])


def upload_to_drive(drive: GoogleDrive,
                    local: str,
                    drive_id: str,
                    ) -> None:
    """Upload content to Google drive.

    Args:
        drive (GoogleDrive): Authenticated drive.
        local_folder (str): Path to local folder or file to upload.
        drive_id (str): Google drive file id upload to.
    """
    
    if os.path.isdir(local):
        subfolder = drive.CreateFile(
            {'title': os.path.basename(local),
             'parents': [{'id': drive_id}],
             'mimeType': 'application/vnd.google-apps.folder'
             }
            )
        subfolder.Upload()
        
        for file in os.listdir(local):
            upload_to_drive(drive,
                            os.path.join(local, file),
                            subfolder['id'])
    
    else:
        file = drive.CreateFile(
            {'title': os.path.basename(local),
             'parents': [{'id': drive_id}]
             })
        file.SetContentFile(local)
        file.Upload()

    return


def clean_drive(drive: GoogleDrive,
                drive_id: str
                ) -> None:
    """Delete all content in Google drive folder.

    Args:
        drive (GoogleDrive): Authenticated drive.
        drive_id (str): Drive folder id to clean.
    """
    drive_folder = drive.ListFile(
        {'q': f"'{drive_id}' in parents and trashed=false",
         'supportsAllDrives': True,
         'includeItemsFromAllDrives': True
         }
        ).GetList()

    for file in drive_folder:
        file.Delete()


def move_drive(drive: GoogleDrive,
               source_id: str,
               tartget_id: str
               ) -> None:
    """Move content in one Google drive folder to the other.

    Args:
        drive (GoogleDrive): Authenticated drive.
        source_id (str): Source Google drive folder id.
        tartget_id (str): Target Google drive folder id.
    """

    drive_folder = drive.ListFile(
        {'q': f"'{source_id}' in parents and trashed=false",
         'supportsAllDrives': True,
         'includeItemsFromAllDrives': True
         }
        ).GetList()

    for file in drive_folder:
        file['parents'] = [{"kind": "drive#parentReference", "id": tartget_id}]
        file.Upload()
