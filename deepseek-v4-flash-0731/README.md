# Deepseek v4 Flash 0731

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s
```

## Start server

```bash
ds4-serve --port 8002 --host 0.0.0.0 # default context is 32768
ds4-serve -c 69632 --port 8002 --host 0.0.0.0 # extended context
ds4-serve -c 131072 --port 8002 --host 0.0.0.0 # extended context
```
