@echo off
cd /d %~dp0

echo ---------------------------------------
echo  初期設定と実行を行っています...
echo ---------------------------------------

:: Pythonが入っているか確認
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Pythonが見つかりません。Pythonをインストールしてください。
    pause
    exit /b
)

:: 仮想環境(venv)がないなら作成
if not exist "venv" (
    echo 仮想環境を作成中...
    python -m venv venv
)

:: 仮想環境を有効化してライブラリインストール
call venv\Scripts\activate
echo ライブラリを確認中...
pip install -r requirements.txt >nul

echo.
echo =======================================
echo  解析スタート！
echo =======================================
python logic.py

echo.
pause