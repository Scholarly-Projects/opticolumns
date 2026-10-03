# Setup

Opticolumns runs on **Python 3.11**. Run all commands from inside the `opticolumns` folder.

## macOS

**1. Install pyenv**

```bash
brew install pyenv xz
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

**2. Install Python 3.11.9**

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

**3. Create the environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Run**

```bash
python script.py                  # OCR: A → B
python review.py                  # review images: B → C
python report.py                  # audit report: A + B → D
```

_To keep your Mac awake during long batches:_

```bash
caffeinate -i python script.py    # prevents idle sleep
caffeinate -di python script.py   # also keeps the display on
```

## Windows

**1. Install pyenv-win** (in PowerShell)

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

Close and reopen PowerShell, then confirm with `pyenv --version`.

**2. Install Python 3.11.9**

```powershell
pyenv install 3.11.9
pyenv local 3.11.9
```

**3. Create the environment and install dependencies**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

_If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again._

**4. Run**

```powershell
python script.py                  # OCR: A → B
python review.py                  # review images: B → C
python report.py                  # audit report: A + B → D
```

_To keep your PC awake during long batches_, Windows has no `caffeinate` equivalent. Note your current settings with `powercfg /query`, then turn sleep off for the run and restore your values afterward:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
python script.py
```

## Next time

Reactivate the environment before running any script:

```bash
source .venv/bin/activate         # macOS
.venv\Scripts\Activate.ps1        # Windows
```