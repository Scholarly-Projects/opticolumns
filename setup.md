# Install pyenv (macOS with Homebrew)

brew install pyenv xz

# Add pyenv to your shell

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Install Python 3.11.9

pyenv install 3.11.9
pyenv local 3.11.9


# Environment

python -m venv .venv
source .venv/bin/activate

# Dependencies

pip install --upgrade pip
pip install -r requirements.txt

# Poppler

brew install poppler

# Run

python script.py
python review.py
