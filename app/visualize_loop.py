import subprocess

while True:
    print("Now running visualization")
    subprocess.call(['python3','./visualize.py',
    '--record_folder','./past_records',
    '--all_records','./cameras.csv'])
