# The virtualenv requires bash.
SHELL = /bin/bash

BASEDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV := $(BASEDIR)keras/

virtualenv:
	test -d $(VENV) || ( virtualenv -p python3 --system-site-packages $(VENV) ) 
	. $(VENV)bin/activate && pip install --upgrade pip
	. $(VENV)bin/activate && pip install -r requirements.txt

all: virtualenv
