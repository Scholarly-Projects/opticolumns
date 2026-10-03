# Setup

Opticolumn_Editor runs on **Python 3.11** with **PyTorch 2.6.0 or newer** (required by CVE-2025-32434). Run all commands from inside the `opticolumn_editor` folder.

# macOS

## Set up

**1. Install pyenv**

```bash
brew install pyenv xz
SHELL_RC="$HOME/.$(basename "$SHELL")rc"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_RC"
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_RC"
echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"
exec "$SHELL"
```

**2. Install Python 3.11.9** (with the compression support Kraken's models need)

```bash
env PYTHON_CONFIGURE_OPTS="--with-liblzma" \
    LDFLAGS="-L$(brew --prefix xz)/lib" \
    CPPFLAGS="-I$(brew --prefix xz)/include" \
    pyenv install 3.11.9
pyenv local 3.11.9
```

**3. Create the environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

_Next time, run `source .venv/bin/activate` before any script._

## Run

_To keep your Mac awake during long batches, put `caffeinate -di` in front of any command, e.g. `caffeinate -di python script.py`._

**1. First pass: OCR** — reads `A`, writes searchable PDFs to `B` and an editable transcription to `C`

```bash
python script.py
```

**2. Edit** the transcription CSV for each document in `C` and save it.

**3. Second pass: apply your edits** — writes corrected PDFs to `D`, using the originals in `A`

```bash
python script.py revised            # uses the edited CSVs in C
python script.py revised_combined   # or: uses the edited .combined.json files in C (text + positions)
```

**4. Refresh the edit files from `D`** — keeps the corrections already in `D` and writes a new CSV and JSON for further rounds of editing

```bash
python script.py post
```

## Audit

**Review images** — the page image on the left, its OCR on the right (output in `E`)

```bash
python review.py B          # first page of each PDF in B
python review.py D          # first page of each PDF in D
python review.py B all      # every page of each PDF in B
python review.py D all      # every page of each PDF in D
```

**Word accuracy report** — true-word counts compared with the originals in `A` (output in `F`)

```bash
python report.py B          # original vs first-pass OCR
python report.py D          # original vs revised OCR
```

## If a file stops the batch

If processing stops with a message ending in `Killed`, a single PDF has used up the computer's memory. The culprit is the file named in the last message before `Killed`.

1. Move that file from the `A` folder to the `review` folder.
2. Run the script again. Files already processed are skipped, so processing picks up where it stopped.

```bash
caffeinate -di python script.py
```

If the file is left in `A`, the script will reach it again and stop at the same point. Files collected in `review` are used to troubleshoot future versions of the script.

# Windows

Kraken, the segmentation model Opticolumn_Editor uses, runs only on Linux and macOS, so on Windows the tool runs inside **WSL** (Windows Subsystem for Linux).

## Set up

**1. Install WSL** (in PowerShell, run as administrator)

```powershell
wsl --install
```

Restart when prompted, then open **Ubuntu** from the Start menu and create a username and password. Run every remaining command in the Ubuntu window.

**2. Install pyenv**

```bash
sudo apt update && sudo apt install -y build-essential curl git libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"
```

**3. Install Python 3.11.9**

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

**4. Create the environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

_Next time, open Ubuntu and run `source .venv/bin/activate` before any script._

_Your Windows drives are available inside Ubuntu under `/mnt/` (for example, `C:\Users\you\Documents\opticolumn_editor` is `/mnt/c/Users/you/Documents/opticolumn_editor`). Processing is faster if the folder lives in your Ubuntu home folder (`~`) instead._

## Run

_To keep your PC awake during long batches_, note your current settings with `powercfg /query` in PowerShell, then turn sleep off for the run and restore your values afterward:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

**1. First pass: OCR** — reads `A`, writes searchable PDFs to `B` and an editable transcription to `C`

```bash
python script.py
```

**2. Edit** the transcription CSV for each document in `C` and save it.

**3. Second pass: apply your edits** — writes corrected PDFs to `D`, using the originals in `A`

```bash
python script.py revised            # uses the edited CSVs in C
python script.py revised_combined   # or: uses the edited .combined.json files in C (text + positions)
```

**4. Refresh the edit files from `D`** — keeps the corrections already in `D` and writes a new CSV and JSON for further rounds of editing

```bash
python script.py post
```

## Audit

**Review images** — the page image on the left, its OCR on the right (output in `E`)

```bash
python review.py B          # first page of each PDF in B
python review.py D          # first page of each PDF in D
python review.py B all      # every page of each PDF in B
python review.py D all      # every page of each PDF in D
```

**Word accuracy report** — true-word counts compared with the originals in `A` (output in `F`)

```bash
python report.py B          # original vs first-pass OCR
python report.py D          # original vs revised OCR
```

## If a file stops the batch

If processing stops with a message ending in `Killed`, a single PDF has used up the computer's memory. The culprit is the file named in the last message before `Killed`.

1. Move that file from the `A` folder to the `review` folder.
2. Run the script again. Files already processed are skipped, so processing picks up where it stopped.

```bash
python script.py
```

If the file is left in `A`, the script will reach it again and stop at the same point. Files collected in `review` are used to troubleshoot future versions of the script.

# Repo Layout

```text
├── A                 source PDFs: add your originals here (never modified)
├── B                 pass 1 output: OCR'd, searchable PDFs
├── C                 pass 1 output / pass 2 input: per-document review CSVs and JSON
├── D                 pass 2 output: corrected PDFs (revised text applied to A/ originals)
├── E                 review.py output
├── F                 report.py output
├── review            PDFs that stopped a batch, set aside for troubleshooting
├── debug_images      auto-saved when a page yields zero detected text lines
├── fonts
│   └── FreeSans.ttf
├── mlmodels
│   └── blla.mlmodel
├── LICENSE
├── README.md
├── notes.md
├── report.py
├── requirements.txt
├── review.py
├── script.py
└── srgb.icc
```