"""Download and extract MovieLens datasets."""
import os
import zipfile
import requests
from tqdm import tqdm

ML1M_URL  = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
ML20M_URL = 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'


def _download_and_extract(url: str, zip_name: str, check_file: str,
                           dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    folder_name = zip_name.replace('.zip', '')
    extract_to  = os.path.join(dest_dir, folder_name)
    zip_path    = os.path.join(dest_dir, zip_name)

    if os.path.isdir(extract_to) and os.path.isfile(
            os.path.join(extract_to, check_file)):
        print(f'[download] {folder_name} already present at {extract_to}')
        return extract_to

    if not os.path.isfile(zip_path):
        print(f'[download] Downloading {folder_name} from {url} ...')
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, desc=zip_name) as bar:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
                bar.update(len(chunk))

    print(f'[download] Extracting {zip_name} to {dest_dir} ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)

    print(f'[download] Done -> {extract_to}')
    return extract_to


def download_ml1m(dest_dir: str = './data') -> str:
    return _download_and_extract(
        ML1M_URL, 'ml-1m.zip', 'ratings.dat', dest_dir)


def download_ml20m(dest_dir: str = './data') -> str:
    return _download_and_extract(
        ML20M_URL, 'ml-20m.zip', 'ratings.csv', dest_dir)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '20m':
        download_ml20m()
    else:
        download_ml1m()
