import os
import urllib.request

def data_from_url(url: str, downloads_dir: str):
    # Split on the rightmost / and take everything on the right side of that
    name = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    filename = os.path.join(downloads_dir, name)
    
    # if folder don't exist
    if not os.path.isdir(downloads_dir):
        os.makedirs(downloads_dir)
        print("created folder : ", downloads_dir)

    # Download the file if it does not exist
    if not os.path.isfile(filename):
        print("File no present, start to download...")
        print("Downloading: " + filename)
        try:
            local_filename, headers = urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(e)
        print("File downloaded")
    else:
        print("File already downloaded")
