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


def getAllPoints(start, end, length):
    x1, y1 = start
    x2, y2 = end
    if x2-x1 == 0:
        return -1
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope*x1
    allPoints = [(x,slope*x+intercept) for x in np.arange(x1, x1+0.25*length, 0.25)]
    return allPoints


def main():
    contour_path = '/media/data_cifs/image_datasets/contours_gilbert_256_INTERPOLATION_OFF'
    make_contours_dir(contour_path)
    curr_radius=4
    min_ecc, max_ecc = get_eccentricity_bounds(curr_radius=curr_radius,
                                               gilb_radius=43.8/2, gilb_min_ecc=2.4,
                                               gilb_max_ecc=8.4)
    print "Eccentricity bounds: %s %s"%(min_ecc ,max_ecc)
    nLinesOnRadius=curr_radius*8
    linesPerDegree = deg2lines(radiusDegrees=curr_radius, nLinesOnRadius=nLinesOnRadius)
    for length in tqdm([15],total=1, desc='Generating multiple length contours'):
        for _ in tqdm(range(200000),desc='Generating contours for length %s'%(length)):
            win = create_window([256,256],monitor='testMonitor',viewOri=0) #np.random.uniform(-180,180))
            circle = draw_circle(win=win,radius=4)
            dot_xys, dot_oris, oriCount = [], [], 0
            dot_xys = [[i,j] for i in np.arange(-4,4,0.5)
                             for j in np.arange(-4,4,0.5)
                             if circle.contains((i,j))]
            ori_orth = np.random.uniform(-180,180)
            start, end = sorted(sample(dot_xys,2))
            start, end = sorted([start,end])
            linePoints = getAllPoints(start, end, length)
            import ipdb; ipdb.set_trace()
            if type(linePoints) == int:
                continue
            while len(dot_oris)<len(dot_xys):
                if dot_xys[oriCount] in linePoints:
                    dot_oris.append(0)
                else:
                    alpha, beta = np.abs(ori_orth+45), np.abs(ori_orth+90) #Driving neighbouring 'distractor' lines to be non-collinear
                    minTheta, maxTheta = min(alpha,beta), max(alpha,beta) #Range of orientation of new 'distractor' line segment
                    ori_orth = np.random.uniform(minTheta, maxTheta)
                    if ori_orth<0:
                        ori_orth += 360 #If angle negative, add 360. <= a = 2.pi + a
                    dot_oris.append(ori_orth)
                oriCount += 1
            dot_stim = visual.ElementArrayStim(
                win=win,
                units="deg",
                fieldSize=(4,4),
                nElements=len(dot_xys),
                elementTex="sin",
                elementMask="gauss",
                sfs=3.1,
                sizes=(0.2,0.5),
                xys=dot_xys,
                oris=dot_oris
            )
            dot_stim.draw()
            win.update()
            import ipdb; ipdb.set_trace()
            win.getMovieFrame()
            win.saveMovieFrames("%s/sample_256_shear_%s_length%s_eccentricity_%s.png"
                                        %(contour_path,shearAngle,length,curr_ecc))
            win.close()

if __name__=="__main__":
    main()
