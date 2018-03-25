#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
from tqdm import tqdm
from random import sample
import os
from random import sample
from matplotlib import pyplot as plt
from gilbert_utils import *
from geometric_transformations import *
from psychopy_utils import *

"""Code to create the snakes dataset for association fields"""
baseSnakeOri = [0]
while(baseSnakeOri[-1]!=135):
    baseSnakeOri += [baseSnakeOri[-1]+45]
baseSnakeOri[0], baseSnakeOri[2] = 90, 0 #Psychopy uses weird angle conventions
#Plausible neighbouring line segment orientations
anglePairs = {
              baseSnakeOri[0]: [baseSnakeOri[1],baseSnakeOri[3],baseSnakeOri[0]],
              baseSnakeOri[1]: [baseSnakeOri[0],baseSnakeOri[2],baseSnakeOri[1]],
              baseSnakeOri[2]: [baseSnakeOri[1],baseSnakeOri[3],baseSnakeOri[2]],
              baseSnakeOri[3]: [baseSnakeOri[0],baseSnakeOri[2],baseSnakeOri[3]]
              }

positionPairs = {
                (baseSnakeOri[0],baseSnakeOri[0]) : [(1,0),(-1,0)],
                (baseSnakeOri[0],baseSnakeOri[1]) : [(1,1),(-1,-1)],
                (baseSnakeOri[0],baseSnakeOri[3]) : [(1,-1),(-1,1)],

                (baseSnakeOri[1],baseSnakeOri[1]) : [(1,1),(-1,-1)],
                (baseSnakeOri[1],baseSnakeOri[0]) : [(1,1),(-1,-1)],
                (baseSnakeOri[1],baseSnakeOri[2]) : [(1,1),(-1,-1)],

                (baseSnakeOri[2],baseSnakeOri[2]) : [(0,1),(0,-1)],
                (baseSnakeOri[2],baseSnakeOri[1]) : [(1,1),(-1,-1)],
                (baseSnakeOri[2],baseSnakeOri[3]) : [(1,-1),(-1,1)],

                (baseSnakeOri[3],baseSnakeOri[3]) : [(1,-1),(-1,1)],
                (baseSnakeOri[3],baseSnakeOri[0]) : [(1,-1),(-1,1)],
                (baseSnakeOri[3],baseSnakeOri[2]) : [(1,-1),(-1,1)]
                }

def getIntercept(x, y, psychopyTheta):
    theta = 90-psychopyTheta
    thetaRad = theta*np.pi/180
    m = np.tan(thetaRad)
    c = y - m*x
    print theta, y, m*x
    return m,c

def inflectionSnakes(win, circle, color=True, size=0.1, length=15, curr_radius=4):
    ori = 90
    min_ecc, max_ecc = get_eccentricity_bounds(curr_radius=curr_radius,
                                                   gilb_radius=43.8/2, gilb_min_ecc=2.4,
                                                   gilb_max_ecc=8.4)
    print min_ecc, max_ecc
    posX = sample(np.arange(-max_ecc, max_ecc, 0.25),1)[0]
    length /= 2
    positionsX = np.arange(posX-0.3*length, posX+0.3*length, 0.3)
    posY = 0
    #posY = sample(np.arange(-curr_radius, curr_radius,0.25),1)[0]
    infl = [positionsX[int(i)] for i in np.linspace(0,len(positionsX)-1,5)]
    infl = infl[1:-1]
    print infl
    prevOri = ori
    prevPos = (positionsX[0], posY)
    allLines = []
    sign = -1
    for posX in positionsX:
        if posX in infl:
            sign *= -1
            ori += sign*45
        m, c = getIntercept(prevPos[0], prevPos[1], ori)
        posY = m*posX + c
        if circle.contains((posX, posY)):
            line = draw_line(win, pos=(posX, posY), contour=False, color=True, size=0.1, ori=ori-45, center=False)
            print posX, posY
            prevPos = (posX, posY)
            allLines.append(line)

    ori_orth = np.random.uniform(-180,180)
    for posX in np.arange(-curr_radius, curr_radius, 0.4):
        for posY in np.arange(-curr_radius, curr_radius, 0.4):
            if circle.contains(posX, posY):
                alpha, beta = np.abs(ori_orth+45), np.abs(ori_orth+90) #Driving neighbouring 'distractor' lines to be non-collinear
                minTheta, maxTheta = min(alpha,beta), max(alpha,beta) #Range of orientation of new 'distractor' line segment
                ori_orth = np.random.uniform(minTheta, maxTheta)
                ln = visual.Line(win=win,pos=(posX, posY),
                                 size=size,ori=np.random.uniform(ori_orth),contrast=1.,
                                 lineColor=(1,1,1),interpolate=True)
                overlaps = [np.sum(np.abs(ln.pos-ln2.pos))<0.3 for ln2 in allLines]
                if not True in overlaps:
                    ln.draw()

def main():
    contour_path = '.'
    make_contours_dir(contour_path)
    win = create_window([256,256],monitor='testMonitor',viewOri=0)
    curr_radius=4
    print "Created window"
    #print "Eccentricity bounds: %s %s"%(min_ecc ,max_ecc)
    nLinesOnRadius=curr_radius*8
    linesPerDegree = deg2lines(radiusDegrees=curr_radius, nLinesOnRadius=nLinesOnRadius)
    circle = draw_circle(win=win,radius=curr_radius)
    for length in tqdm([1],total=1, desc='Generating multiple length contours'):
        for _ in tqdm(range(100),desc='Drawing snakes..'):
            win.viewOri = np.random.uniform(-180,180)
            draw_grating_stim(win, (0,0), 0.1)
            positions = [(j,i) for i in np.arange(-curr_radius*2,curr_radius*2,0.25) for j in np.arange(-curr_radius*2,curr_radius*2,0.25)]
            positions = np.array(positions).reshape((curr_radius*16,curr_radius*16,2))
            inflectionSnakes(win, circle)
            win.flip()
            import ipdb; ipdb.set_trace()
            win.getMovieFrame()
            win.saveMovieFrames("%s/curves_%s.png"%(contour_path,length))
    win.close()

if __name__=="__main__":
    main()
