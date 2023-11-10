import os
import requests
import subprocess
import shutil

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def format_drive(drive):
    confirm = input(f"Are you sure you want to format the drive {drive}? This will erase all data. Type 'yes' to confirm: ")
    if confirm.lower() == 'yes':
        os.system(f'format {drive} /FS:NTFS /Q /V:WinPython')
    else:
        print("Formatting cancelled.")
        exit()

def copy_contents(source_folder, dest_drive):
    for item in os.listdir(source_folder):
        s = os.path.join(source_folder, item)
        d = os.path.join(dest_drive, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def main():
    # User inputs the drive
    drive = input("Enter the drive letter for the Pendrive (e.g., D:): ")

    # Format the Pendrive
    format_drive(drive)

    # Download and install WinPython
    winpython_url = "https://github.com/winpython/winpython/releases/download/7.0.20230928/Winpython64-3.11.5.0.exe"
    winpython_exe = download_file(winpython_url, os.getcwd())
    subprocess.run([winpython_exe, '/SILENT'])

    # Download the model
    model_url = "https://gpt4all.io/models/gguf/mistral-7b-openorca.Q4_0.gguf"
    download_file(model_url, os.path.join(os.getcwd(), 'models'))

    # Copy contents to Pendrive
    copy_contents('driver', drive)

    # Change directory to Pendrive and run scripts
    os.chdir(drive)
    subprocess.run(['python', 'ingest.py'])
    subprocess.run(['python', 'run.py'])

    # Start the Streamlit app
    subprocess.run(['streamlit', 'run', 'app.py'])

if __name__ == "__main__":
    main()
