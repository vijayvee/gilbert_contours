#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
from tqdm import tqdm
import os
from random import sample
from matplotlib import pyplot as plt

def shear(points,shearX):
    points_ = points.reshape(-1,2)
    shear_matrix = np.array([[1,0],[shearX,1]])
    new_points = points_.dot(shear_matrix)
    new_points = new_points.reshape(points.shape)
    return new_points

def get_orthogonal_angles(nrows,ncols):
    angles = np.random.uniform(size=(nrows,ncols),low=-1,high=1)
    angles_o = scipy.linalg.orth(angles)
    return angles_o

def findAngle(posA, posB):
    aX, aY = posA
    bX, bY = posB
    m = (bY-aY)/(bX-aX)
    angle = np.arctan(m)*360/np.pi
    return angle

def rotate(pointsToRot, angle, contourPosition):
    if type(pointsToRot)==list:
        pointsToRot = np.array(pointsToRot)
    nPoints = len(pointsToRot)
    pointsToRot[:,0] -= contourPosition[0]
    pointsToRot[:,1] -= contourPosition[1]
    pointsToRot = np.hstack((pointsToRot,np.ones((nPoints,1))))
    rot = np.array([[np.cos(angle),-np.sin(angle), contourPosition[0]],[np.sin(angle),np.cos(angle),contourPosition[1]],[0,0,1]])
    rotated = rot.dot(pointsToRot.T).T
    rotated = rotated[:,[0,1]]
    rotated = np.round(rotated).astype(np.int32)
    m,b = np.polyfit(rotated[:,0],rotated[:,1],1)
    rotatedSmoothY = np.round(m*rotated[:,0] + b)
    rotated[:,1] = rotatedSmoothY
    return rotated
