# Create a virtual environment
python -m venv ditto_venv

# Activate the virtual environment
.\ditto_venv\Scripts\Activate

# Install modules from requirements.txt
pip install -r requirements.txt

Write-Output "ditto_venv/" > .\.gitignore
Write-Output "aditi ne palak charaya , which is insain to anirudh" > .\message.txt
