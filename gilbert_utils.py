#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
from tqdm import tqdm
import os
from random import sample
from matplotlib import pyplot as plt
from geometric_transformations import *
"""Utils for plotting contours"""

def make_contours_dir(contour_path):
    """Create directory to save generated contours"""
    if not os.path.exists(contour_path):
        os.mkdir(contour_path)

def get_contour_center(a_contour, curr_radius):
    """Get position of the center of generated contour"""
    rX, rY = curr_radius*np.cos(a_contour), curr_radius*np.sin(a_contour)
    print rX, rY, curr_radius, a_contour
    return rX, rY

def deg2lines(radiusDegrees, nLinesOnRadius):
    """Convert between visual degrees to contour grid units"""
    linesPerDegree = nLinesOnRadius/radiusDegrees
    return linesPerDegree

def get_eccentricity_bounds(curr_radius, gilb_radius, gilb_min_ecc, gilb_max_ecc):
    """Convert proportionally, the eccentricity bounds used in the physiology paper to match our current radius"""
    min_ecc = gilb_min_ecc*curr_radius/gilb_radius
    max_ecc = gilb_max_ecc*curr_radius/gilb_radius
    return min_ecc, max_ecc

def getContourOrientation(shearAngle):
    """Function to obtain angle of orientation for contour line segments
       Found the following orientations to work best for aligning contours by trial-and-error
       TODO: Compute angles based on shear more systematically"""
    if shearAngle>0:
        ori = (shearAngle*15.9)
    else:
        ori=-np.abs(shearAngle)*45.9
    return ori

def fillIncludeContour(nRows,nCols,pos,length=5,a_contour=0):
    """Function to compute positions of contour line segments in contour grid"""
    includeContour = np.zeros((nRows,nCols))
    includeContour[pos[0],pos[1]] = 1
    bounds = (length+1)/2
    pointsToRot = [[pos[0]+i,pos[1]] for i in range(-bounds,bounds)]
    pointsRotated = rotate(pointsToRot,np.pi/2-a_contour,pos) #rotate 90-a_contour degrees about contourPosition, making contour path tangential to circle
    rotated = np.round(pointsRotated).astype(np.int32)
    for i in range(len(rotated)):
        includeContour[rotated[i,0],rotated[i,1]] = 1
    return includeContour
