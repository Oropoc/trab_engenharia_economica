@echo off
setlocal

cd /d "%~dp0"
set "PYTHON_EXE=C:\Python314\python.exe"
set "PYTHONPATH=C:\ai_env\site-packages"

if not exist "%PYTHON_EXE%" (
    echo ERRO: Python nao encontrado em "%PYTHON_EXE%".
    echo Ajuste PYTHON_EXE no arquivo rodar_tudo.bat.
    pause
    exit /b 1
)
if not exist "%PYTHONPATH%" (
    echo ERRO: Bibliotecas nao encontradas em "%PYTHONPATH%".
    echo Ajuste PYTHONPATH no arquivo rodar_tudo.bat ou instale requirements.txt.
    pause
    exit /b 1
)

echo ======================================================
echo Executando preparacao dos dados...
echo ======================================================
"%PYTHON_EXE%" .\scripts\01_preparacao_dados.py
if errorlevel 1 (
    echo ERRO na preparacao dos dados.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Executando analise exploratoria...
echo ======================================================
"%PYTHON_EXE%" .\scripts\02_analise_exploratoria.py
if errorlevel 1 (
    echo ERRO na analise exploratoria.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Executando classificacao multiclasse...
echo ======================================================
"%PYTHON_EXE%" .\scripts\03_classificacao_multiclasse.py
if errorlevel 1 (
    echo ERRO na classificacao multiclasse.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Executando classificacao com ajuste de hiperparametros...
echo ======================================================
"%PYTHON_EXE%" .\scripts\04_classificacao_otimizada.py
if errorlevel 1 (
    echo ERRO na classificacao otimizada.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Executando regressao de contagens por celula...
echo ======================================================
"%PYTHON_EXE%" .\scripts\05_regressao_contagem.py
if errorlevel 1 (
    echo ERRO na regressao de contagens.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Identificando hotspots espaciais...
echo ======================================================
"%PYTHON_EXE%" .\scripts\06_hotspots_espaciais.py
if errorlevel 1 (
    echo ERRO na identificacao de hotspots.
    pause
    exit /b 1
)

echo.
echo Tudo concluido com sucesso!
pause
