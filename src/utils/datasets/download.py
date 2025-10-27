import yaml
import tarfile
import zipfile
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

def _requests_session_with_retries(total_retries=5, backoff=0.5):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def download_file_stream(url, dest_path, chunk_size=8192):
    session = _requests_session_with_retries()
    temp_path = dest_path + ".part"
    resume_header = {}
    if os.path.exists(temp_path):
        existing = os.path.getsize(temp_path)
        resume_header = {"Range": f"bytes={existing}-"}
    else:
        existing = 0

    with session.get(url, stream=True, headers=resume_header, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        if total is not None:
            total = int(total) + existing
        mode = "ab" if existing else "wb"
        with open(temp_path, mode) as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest_path), initial=existing) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    os.replace(temp_path, dest_path)
    return dest_path

def download_datasets(url, path_download, path_dest):
    os.makedirs(path_download, exist_ok=True)
    os.makedirs(path_dest, exist_ok=True)

    file_name = os.path.join(path_download, os.path.basename(urllib.parse.urlparse(url).path) or "download")
    print(f"Baixando arquivo {file_name}...")

    try:
        download_file_stream(url, file_name)
        print("Download concluído!")
    except Exception as e:
        print(f"Erro no download: {e}")
        return

    # Extração
    print("Extraindo arquivo...")
    
    if file_name.endswith('.tar.gz'):
        with tarfile.open(file_name, 'r:gz') as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extraindo TAR.GZ"):
                tar.extract(member, path=path_dest)

    elif file_name.endswith('.zip'):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Extraindo ZIP"):
                zip_ref.extract(member, path=path_dest)

    else:
        print("Tipo de arquivo não suportado para extração.")

    print("Arquivo extraído com sucesso!")
    os.remove(file_name)
    print("Arquivo compactado removido.")

if __name__ == "__main__":
    with open('/home/c.oliveira25/Desktop/DeepLearning/config/dataset.yaml', 'r') as file:
        data = yaml.safe_load(file)

    for dataset in data['datasets']:
        try: 
            dest_download = data['datasets'][dataset]['paths']['raw']
            dest_folder = data['datasets'][dataset]['paths']['processed']
            url = data['datasets'][dataset]['link_download']
            download_datasets(url, dest_download, dest_folder)
        except KeyError:
            print(f"Nenhum link de download para o dataset {dataset}, pulando...")

        
