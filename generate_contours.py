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
"""Code to recreate the snakes dataset for association fields
   Testing multiple tricks to generate stimuli that are highly similar to https://www.physiology.org/doi/pdf/10.1152/jn.00289.2002"""

def draw_lines_row(win, circle, positions, color=False, size=0.13, shearAngle=0.3, length=3, contourPosition=(10,20),a_contour=0):
    """Function to draw the main contour line segments."""
    
    includeContour = np.zeros((positions.shape[0],positions.shape[1]))
    #Set includeContour[i,j]=1. if position i,j of contour grid is along the contour path
    includeContour = fillIncludeContour(includeContour, contourPosition, length=length, a_contour=a_contour)
    if shearAngle!=0:
        positions_ = shear(positions,shearX=shearAngle) #Shearing to adjust intra-contour
    else:
        positions_ = positions #Shear=0 implies no shearing
    oriContour = -1
    ori_orth = np.random.uniform(-180,180) #Choose random starting angle for distractor line segments
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            #Main loop, align elements along the contour and randomly orient other elements in contour grid
            pos = positions_[i,j,:]
            rect = visual.Rect(win,pos=pos,width=0.2,height=0.2,fillColor=(1,1,1))
            print rect.vertices
            rect.draw()
            continue
            pos = positions_[i,j,:]
            if circle.contains(pos,units='deg'):
                if includeContour[i,j]==1:
                    if color: #Coloring line segments along the generated contour
                        if i==contourPosition[0] and j==contourPosition[1]: #Contour center with different color
                            lineColor=(-1,1,-1)
                        else:
                            lineColor=(1,-1,-1) #Non-center elements of contour with different color
                    else:
                        lineColor=(1,1,1) #No color difference between contours and distractors
                    #Found the following orientations to work best for aligning contours by trial-and-error
                    #TODO: Compute angles based on shear more systematically
                    if shearAngle>0:
                        ori = (shearAngle*15.9)
                    else:
                        ori=-np.abs(shearAngle)*45.9
                else:
                    lineColor=(1,1,1)
                    alpha, beta = np.abs(ori_orth+45), np.abs(ori_orth+90) #Driving neighbouring 'distractor' lines to be non-collinear
                    minTheta, maxTheta = min(alpha,beta), max(alpha,beta) #Range of orientation of new 'distractor' line segment
                    ori_orth = np.random.uniform(minTheta, maxTheta)
                    if ori_orth<0:
                        ori_orth += 360 #If angle negative, add 360. <= a = 2.pi + a
                    ori = ori_orth
                ln = visual.Line(win, pos=pos, size=size, ori=ori,lineColor=lineColor)
                ln.draw()

def main():
    contour_path = '/media/data_cifs/image_datasets/contours_gilbert_600'
    make_contours_dir(contour_path)
    win = create_window([600,600],monitor='testMonitor')
    curr_radius=8
    print "Created window"
    min_ecc, max_ecc = get_eccentricity_bounds(curr_radius=curr_radius, gilb_radius=43.8/2, gilb_min_ecc=3.4, gilb_max_ecc=6.4)
    print "Eccentricity bounds: %s %s"%(min_ecc ,max_ecc)
    linesPerDegree = deg2lines(radiusDegrees=curr_radius, nLinesOnRadius=32)
    for shearAngle in tqdm([0],total=1,desc='Generating multiple spacing contours'): #, 0.7, 1.3]:
        print "Shear angle: ", shearAngle
	for length in tqdm([21,18,15],total=3, desc='Generating multiple length contours'):
            for _ in tqdm(range(20),desc='Generating contours for length %s'%(length)):
                curr_ecc = np.random.uniform(min_ecc, max_ecc)
                ecc_lines = curr_ecc*linesPerDegree
                a_contour = np.random.uniform(0,np.pi/2)
                pos = get_contour_center(a_contour, curr_ecc)
                pos = 30+int(linesPerDegree*(pos[1])), 30-int(linesPerDegree*(pos[0]))
                print "Current eccentricity: %s, current angle: %s, current center contour: (%s,%s)" %(curr_ecc, a_contour*180/np.pi, pos[0],pos[1])
                circle = draw_circle(win=win,radius=8)
                positions = [(j,i) for i in np.arange(-15,15,0.5) for j in np.arange(-15,15,0.5)]
                positions = np.array(positions).reshape((60,60,2))
                draw_lines_row(win, circle, positions,color=True,
                                   length=length,shearAngle=shearAngle,
                                   contourPosition=pos, a_contour=a_contour)
                draw_fixation(win, 0, 0)
                win.update()
                import ipdb; ipdb.set_trace()
                win.getMovieFrame()
                win.saveMovieFrames("%s/sample_shear_%s_length%s_eccentricity_%s.png"
                                            %(contour_path,shearAngle,length,curr_ecc))
    win.close()

if __name__=="__main__":
    main()
