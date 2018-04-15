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
        os.makedirs(contour_path)

def get_contour_center(a_contour, curr_radius):
    """Get position of the center of generated contour"""
    rX, rY = curr_radius*np.cos(a_contour), curr_radius*np.sin(a_contour)
    return rX, rY

def get_gaussian_1d(sigma, points, order):
    points = np.float_(points)
    variance = sigma*sigma
    op1 = -points/variance
    points = points*points
    denom = 2*np.pi*variance
    probs = np.exp(-points/(2*variance))/np.sqrt(denom)
    if order==1:
     probs = op1*probs
    return probs

def get_gaussian_derivatives(size, ori):
    #RECHECK THIS FUNCTION!
    k = size/2
    orgpts = np.array([(i,j) for i,j in zip(np.arange(-k,k+1),np.arange(-k,k+1))])
    gx = get_gaussian_1d(3,orgpts[:,0],0)
    gy = get_gaussian_1d(1,orgpts[:,1],1)
    kern = np.outer(gx, gy)
    return kern

def drawSnakes(win, circle, positions, color=True, size=0.1, length=15, contourPosition=(32,32)):
    nRows, nCols = positions.shape[0], positions.shape[1]
    #posX, posY = sample(range(nRows),1)[0], sample(range(nCols),1)[0]
    prevPos = positions[contourPosition]
    prevOri = sample(baseSnakeOri,1)[0]
    print "Starting at ",prevPos
    fillMap = [(i,j) for i in np.arange(-4,4,0.25)
                     for j in np.arange(-4,4,0.25)
                     if circle.contains((i,j))]
    for i in range(length):
        currOri = sample(anglePairs[prevOri],1)[0]
        currPosPair = positionPairs[prevOri,currOri][0]
        print currPosPair
        currPosPair = [0.25*currPosPair[0], 0.25*currPosPair[1]]
        currPos = prevPos[0]+currPosPair[0], prevPos[1]+currPosPair[1]
        if circle.contains(currPos):
            if currPos in fillMap:
                fillMap.remove(currPos)
            print "PrevOri:%s CurrOri:%s"%(prevOri, currOri)
            draw_line(win, pos=currPos, contour=True, color=False, size=0.1, ori=currOri-45, center=False)
            prevPos, prevOri = currPos, currOri
    ori_orth = np.random.uniform(-180,180)
    for pos in fillMap:
        alpha, beta = np.abs(ori_orth+45), np.abs(ori_orth+90) #Driving neighbouring 'distractor' lines to be non-collinear
        minTheta, maxTheta = min(alpha,beta), max(alpha,beta) #Range of orientation of new 'distractor' line segment
        ori_orth = np.random.uniform(minTheta, maxTheta)
        pos = (pos[0] + np.random.uniform(-0.1,0.1), pos[1] + np.random.uniform(-0.1,0.1))
        draw_line(win, pos=pos, contour=False, color=False, size=0.1, ori=ori_orth, center=False)


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
