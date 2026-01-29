@echo off
echo starting venv
call .venv\Scripts\activate
echo starting indexing
python server_api.py -p 8765
pause