#!/usr/bin/env bash

apt-get update
apt-get install -y chromium wget gnupg unzip curl

# Make sure Chrome is executable for u-chromedriver
ln -s /usr/bin/chromium /usr/bin/google-chrome
