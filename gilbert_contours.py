#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
from tqdm import tqdm
import os
from random import sample
from matplotlib import pyplot as plt
from gilbert_utils import *
from geometric_transformations import *
from psychopy_utils import *

"""Code to create the snakes dataset for association fields"""

def fillIncludeContour_nontan(nRows,nCols, pos, length=5):
    #Include contours by relaxing the tangential contour path condition
    includeContour = np.zeros((nRows,nCols))
    if length == 0:
        return includeContour
    includeContour[pos[0],pos[1]] = 1
    bounds = (length+1)/2
    for i in range(1,bounds):
        includeContour[pos[0]+i,pos[1]+i] = 1
        includeContour[pos[0]-i,pos[1]-i] = 1
    return includeContour


def draw_lines_row(win, circle, positions, color=False, size=0.1, shearAngle=0.3, length=3, contourPosition=(10,20)):
    """Function to draw the main contour line segments."""
    #Set includeContour[i,j]=1. if position i,j of contour grid is along the contour path
    includeContour = fillIncludeContour_nontan(nRows=positions.shape[0],
                                               nCols=positions.shape[1],
                                               pos=contourPosition, length=length)
    positions_ = shear(positions,shearX=shearAngle) #Shearing to adjust intra-contour
    oriContour,contour,center = -1,False,False
    ori_orth = np.random.uniform(-180,180) #Choose random starting angle for distractor line segments
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            #Main loop, align elements along the contour and randomly orient other elements in contour grid
            pos = positions_[i,j,:]
            if circle.contains(pos,units='deg'):
                if includeContour[i,j]==1:
                    ori = getContourOrientation(shearAngle)
                    contour=True
                    if np.all(contourPosition==(i,j)):
                        center=True
                else:
                    alpha, beta = np.abs(ori_orth+45), np.abs(ori_orth+90) #Driving neighbouring 'distractor' lines to be non-collinear
                    minTheta, maxTheta = min(alpha,beta), max(alpha,beta) #Range of orientation of new 'distractor' line segment
                    ori_orth = np.random.uniform(minTheta, maxTheta)
                    if ori_orth<0:
                        ori_orth += 360 #If angle negative, add 360. <= a = 2.pi + a
                    ori = ori_orth
                    contour=False
                    center=False
                    ori=np.random.uniform(-180,180)
                draw_line(win, pos=pos, contour=contour, color=color, size=size, ori=ori, center=center)


def main():
    contour_path = '/media/data_cifs/image_datasets/contours_gilbert_256_length_0'
    make_contours_dir(contour_path)
    win = create_window([256,256],monitor='testMonitor')
    curr_radius=4
    print "Created window"
    min_ecc, max_ecc = get_eccentricity_bounds(curr_radius=curr_radius,
                                               gilb_radius=43.8/2, gilb_min_ecc=2.4,
                                               gilb_max_ecc=8.4)
    print "Eccentricity bounds: %s %s"%(min_ecc ,max_ecc)
    nLinesOnRadius=curr_radius*8
    linesPerDegree = deg2lines(radiusDegrees=curr_radius, nLinesOnRadius=nLinesOnRadius)
    for shearAngle in tqdm([0.7],total=1,desc='Generating multiple spacing contours'):
        shearAngle = np.random.uniform(low=-0.7, high=0.7)
        print "Shear angle: ", shearAngle
	for length in tqdm([0],total=1, desc='Generating multiple length contours'):
            for _ in tqdm(range(140000),desc='Generating contours for length %s'%(length)):
                shearAngle = np.random.uniform(low=-0.7, high=0.7)
                curr_ecc = np.random.uniform(min_ecc, max_ecc)
                ecc_lines = curr_ecc*linesPerDegree
                a_contour = np.random.uniform(0,np.pi/2)
                pos = get_contour_center(a_contour, curr_ecc)
                pos = nLinesOnRadius+int(linesPerDegree*(pos[1])), nLinesOnRadius-int(linesPerDegree*(pos[0]))
                circle = draw_circle(win=win,radius=curr_radius)
                positions = [(j,i) for i in np.arange(-curr_radius*2,curr_radius*2,0.25) for j in np.arange(-curr_radius*2,curr_radius*2,0.25)]
                positions = np.array(positions).reshape((curr_radius*16,curr_radius*16,2))
                draw_lines_row(win, circle, positions,color=False,
                                   length=length,shearAngle=shearAngle,
                                   contourPosition=pos)
                win.update()
                win.getMovieFrame()
                win.saveMovieFrames("%s/sample_256_shear_%s_length%s_eccentricity_%s.png"
                                            %(contour_path,shearAngle,length,curr_ecc))
    win.close()

if __name__=="__main__":
    main()
