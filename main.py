import subprocess
import os

if __name__ == "__main__":
    subprocess.run(["python", "-m", "streamlit", "run", os.path.abspath("app.py")])
