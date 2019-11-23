"""
CADET-Process
========

CADET-Process is a Python package for modelling, simulating and optimizing
advanced chromatographic systems. It serves as an inteface for CADET, but also for other solvers.
"""
# Basic functionality
from CADETProcess.CADETProcessError import *

import CADETProcess.common

import CADETProcess.processModel
from CADETProcess.processModel import unitOperation, binding
from CADETProcess.processModel import FlowSheet, Process

import CADETProcess.optimization

import CADETProcess.simulation

import CADETProcess.fractionation
