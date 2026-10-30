"""Download the unmodified UCI red wine data (CC BY 4.0)."""
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/186/wine+quality.zip'

if __name__ == '__main__':
    with urlopen(URL, timeout=60) as response:
        archive = ZipFile(io.BytesIO(response.read()))
    (ROOT / 'data').mkdir(exist_ok=True)
    for name in ['winequality-red.csv', 'winequality.names']:
        (ROOT / 'data' / name).write_bytes(archive.read(name))
    print('Downloaded original red wine CSV and dataset description from UCI.')
