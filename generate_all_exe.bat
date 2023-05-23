cd .\CF_item
pyinstaller -c -F main.py -p config.py -p utils.py -p cf_class.py -p user.py
move .\dist\main.exe .\
rd /s/q .\build
rd /s/q .\dist
del main.spec
pause
cd ..
cd .\LFM
pyinstaller -c -F main.py -p config.py -p LFM.py
move .\dist\main.exe .\ 
rd /s/q .\build
rd /s/q .\dist
del main.spec
pause
cd ..
cd .\Regression
pyinstaller -c -F main.py -p config.py -p dataloader.py -p modelzoo.py -p utils.py -p comp.py
move .\dist\main.exe .\
rd /s/q .\build
rd /s/q .\dist
del main.spec
pause