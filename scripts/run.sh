#!/bin/bash

ssh Thien@192.168.1.2 "powershell -Command \"Set-Location C:\workspace\slm_evaluator; .venv\Scripts\python.exe scripts\remote_latency_10runs.py\"" 2>&1 | grep -v WARNING | grep -v "store now" | grep -v "\\\\store"