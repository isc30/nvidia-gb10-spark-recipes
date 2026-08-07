# Deepseek v4 Flash 0731

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s
ds4-serve --port 8000 --host 0.0.0.0
```

## Start server

```bash
ds4-serve --port 8000 --host 0.0.0.0 # default context 32768
ds4-serve -c 69632 --port 8000 --host 0.0.0.0 # extended context
```
