#!/bin/bash
        python train.py --logging-dir ./snapshot/
        python predict.py --logging-dir ./snapshot/
