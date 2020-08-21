#!/usr/bin/env bash

rm -rf build
python setup.py clean && python setup.py build develop