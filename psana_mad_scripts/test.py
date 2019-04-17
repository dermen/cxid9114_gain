import subprocess

for i in range(2):
    subprocess.call("ls -lrt", shell=True)
