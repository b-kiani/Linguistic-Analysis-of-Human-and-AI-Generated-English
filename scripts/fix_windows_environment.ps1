# Run this from: E:\Papers\ai_linguistics_project
# PowerShell only.

Set-Location E:\Papers\ai_linguistics_project

# Activate existing venv. Do NOT recreate it while it is active.
.\.venv\Scripts\Activate.ps1

# Optional but useful.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Download and prepare HC3 / BAWE / TOEFL11 note.
python scripts\download_datasets.py --overwrite
python scripts\prepare_hc3_binary_dataset.py
