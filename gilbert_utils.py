#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
from tqdm import tqdm
import os
from random import sample
from matplotlib import pyplot as plt
from geometric_transformations import *


def make_contours_dir(contour_path):
    if not os.path.exists(contour_path):
        os.mkdir(contour_path)
        
def get_contour_center(a_contour, curr_radius):
    rX, rY = curr_radius*np.cos(a_contour), curr_radius*np.sin(a_contour)
    return rX, rY

def deg2lines(radiusDegrees, nLinesOnRadius):
    linesPerDegree = nLinesOnRadius/radiusDegrees
    return linesPerDegree

def get_eccentricity_bounds(curr_radius, gilb_radius, gilb_min_ecc, gilb_max_ecc):
    min_ecc = gilb_min_ecc*curr_radius/gilb_radius
    max_ecc = gilb_max_ecc*curr_radius/gilb_radius
    return min_ecc, max_ecc

def fillIncludeContour(includeContour, pos, length=5, a_contour=0):
    includeContour[pos[0],pos[1]] = 1
    bounds = (length+1)/2
    pointsToRot = [[pos[0]+i,pos[1]] for i in range(-bounds,bounds)]
    pointsRotated = rotate(pointsToRot,np.pi/2-a_contour,pos) #rotate 90-a_contour degrees about contourPosition, making contour path tangential to circle
    rotated = np.round(pointsRotated).astype(np.int32)
    for i in range(len(rotated)):
        includeContour[rotated[i,0],rotated[i,1]] = 1
    return includeContour
