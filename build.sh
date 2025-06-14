#!/usr/bin/env bash
apt-get update
apt-get install -y curl unzip
python -m playwright install
