@echo off


echo 正在清理LaTex编译中间文件，dvi、ps 文件也将删除，按任意键 执行  

echo. & pause 

del /f /s /q  *.aux
del /f /s /q  *.bbl
del /f /s /q  *.log
del /f /s /q  *.blg
del /f /s /q  *.dvi
del /f /s /q  *.synctex.gz
del /f /s /q  *.bak
del /f /s /q  *.sav
del /f /s /q  *.out
del /f /s /q  *.spl
del /f /s /q  *.nav
del /f /s /q  *.snm
del /f /s /q  *.vrb
del /f /s /q  *.toc
del /f /s /q  *.un~
del /f /s /q  *.bat~
del /f /s /q  *.txt~
del /f /s /q  *.tex~
del /f /s /q  *.bib~

 

echo 完成清理 LaTex编译中间文件 ！
echo. & pause 
