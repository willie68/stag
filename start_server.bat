@echo off
echo starting venv
call .venv\Scripts\activate
echo starting indexing
python server.py -p 8765
pause