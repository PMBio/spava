from tqdm import tqdm
import requests
import tifffile
import tempfile
import os
import math

OME_AND_MASKS_URL = 'https://zenodo.org/record/3518284/files/OMEandSingleCellMasks.zip'
OME_AND_MASKS_HASH = '777f8a59da4f4efc2fcd7149565dd191'
METADATA_URL = 'https://zenodo.org/record/3518284/files/SingleCell_and_Metadata.zip'
METADATA_HASH = '157756ca703e6cfc73377c60d39dcb19'


def download_file(url, output_dir):
    local_filename = os.path.join(output_dir, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            bar = tqdm(total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()
    return local_filename


with tempfile.TemporaryDirectory() as tempdir:
    print(tempdir)
    download_file(METADATA_URL, output_dir='.')
    # f = tempfile.TemporaryFile()
    # f.write('something on temporaryfile')
    # f.seek(0) # return to beginning of file
    # print f.read() # reads data back from the file
    # f.close() # temporary file is automatically deleted here
