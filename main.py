from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class GoogleManager:
    def __init__(self):
        self.__auth = GoogleAuth()
        self.__auth.LocalWebserverAuth()
        self.drive = GoogleDrive(self.__auth)

    def list_folder(self, parent):
        file_list = []
        query = {'q': f"{parent} in parents and trashed=false"}
        for f in self.drive.ListFile(query).GetList():
            print("XD")
            if f['mimeType'] == 'application/vnd.google-apps.folder':
                file_list.append({"id": f['id'], "title": f['title'],
                                  "list": self.list_folder(f['id'])})
            else:
                file_list.append(f['title'])
        return file_list


def main():
    google_manager = GoogleManager()
    print("'dupa'")
    google_manager.list_folder("'root'")
    print("dupa")

    # Auto-iterate through all files that matches this query
    # file_list = drive.ListFile(
    #     {'q': "'root' in parents and trashed=false"}).GetList()
    # for file_list in drive.ListFile({'q': 'trashed=false', 'maxResults': 10}):
    #     print('Received %s files from Files.list()' % len(file_list))  # <= 10
    #     for file in file_list:
    #         print(f"title: {file['title']}")


if __name__ == "__main__":
    main()
