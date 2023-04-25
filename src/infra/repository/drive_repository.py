from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import yaml
import json
import os


class DriveRepository:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = f"{base_path}/../../../resource"
        settings_file = f"{self.resource_path}/settings.yaml"
        settings = open(settings_file, "r")
        client_secret = open(f"{self.resource_path}/client_secret.json", "r")
        client_secret_json = json.load(client_secret)

        settings_yaml = yaml.load(settings, Loader=yaml.FullLoader)

        settings_yaml["client_config"]["client_id"] = client_secret_json["installed"]["client_id"]
        settings_yaml["client_config"]["client_secret"] = client_secret_json["installed"]["client_secret"]
        settings_yaml["save_credentials_file"] = f"{self.resource_path}/credentials.json"

        self.folder_id = settings_yaml["base_folder_id"]

        with open(settings_file, "w") as yaml_file:
            yaml_file.write(yaml.dump(settings_yaml, default_flow_style=False))

        g_auth = GoogleAuth(settings_file=f"{self.resource_path}/settings.yaml")
        g_auth.LocalWebserverAuth()
        self.drive = GoogleDrive(g_auth)

    def recursive_upload(self, path_actual: str, parent_id: str | None):
        if parent_id is None:
            parent_id = self.folder_id

        if os.path.isdir(path_actual):
            folder_name = path_actual.split("\\")[-1]
            level_path = len(path_actual.split("\\"))

            parent_id = self.create_folder(folder_name, parent_id)

            for path, _, files in os.walk(path_actual):
                for file in files:
                    file_path = os.path.join(path, file)
                    if (len(file_path.split("\\")) - 1) == level_path:
                        self.recursive_upload(file_path, parent_id)

                if len(path.split("\\")) == (level_path + 1):
                    self.recursive_upload(path, parent_id)

        elif os.path.isfile(path_actual):
            print(f"Uploading: {path_actual}...")
            file_name = path_actual.split("\\")[-1]
            self.upload_file(file_name, path_actual, parent_id)

    def get_folder_id(self, folder_name: str, parent_id: str) -> str:
        file_list = self.drive.ListFile({'q': f"'{parent_id}' in parents and trashed=false"}).GetList()
        for file in file_list:
            if file['title'] == folder_name:
                return file['id']

        return self.folder_id

    def create_folder(self, folder_name: str, parent_folder_id: str) -> str:
        if self.get_folder_id(folder_name, parent_folder_id) == self.folder_id:
            file_metadata = {
                'title': folder_name,
                'parents': [{'id': parent_folder_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }

            folder = self.drive.CreateFile(file_metadata)
            folder.Upload()

        return self.get_folder_id(folder_name, parent_folder_id)

    def upload_file(self, file_name: str, file_path: str, parent_folder_id: str) -> str:
        if self.get_folder_id(file_name, parent_folder_id) == self.folder_id:
            file_metadata = {
                'title': file_name,
                'parents': [{'id': parent_folder_id}],
            }

            file = self.drive.CreateFile(file_metadata)
            file.SetContentFile(file_path)
            file.Upload()

        return self.get_folder_id(file_name, parent_folder_id)
