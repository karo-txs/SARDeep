from src.infra.repository.drive_repository import DriveRepository

drive = DriveRepository()

if __name__=="__main__":
    drive.create_folder("abc/bca")