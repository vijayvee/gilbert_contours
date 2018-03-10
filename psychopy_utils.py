#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
from tqdm import tqdm
import os
from random import sample
from matplotlib import pyplot as plt

def create_window(window_size, monitor):
    """Function to create a window to plot stimuli"""
    mywin = visual.Window([window_size[0],window_size[1]],
                          color=(-1,-1,-1),monitor=monitor,
                          units="deg")
    return mywin

def draw_line(win, pos, contour, color, size=3., ori=0, center=False):
    """Function to draw a line"""
    lineColor=(1,1,1)
    if color: #Coloring line segments along the generated contour
        if contour: #Contour center with different color
            lineColor=(1,-1,-1) #Plot elements of contour with different color
    else:
        lineColor=(1,1,1) #No color difference between contours and distractors
    if center:
        lineColor=(-1,1,-1)
    ln = visual.Line(win=win,pos=pos,size=size,ori=ori,lineColor=lineColor)
    ln.draw()

def draw_grating_stim(win, pos, size,ori=0, lineColor=(1,1,1)):
    """Function to draw a grating stimulus"""
    stim = visual.GratingStim(win=win, pos=pos, size=(0.3,1.0),
                              tex='sin', mask='cross', units='deg',
                              sf=1.5, ori=ori, color=lineColor)
    return stim

def draw_circle(win,radius,lineWidth=3.,edges=48):
    """"Function to draw a circle"""
    crcl = visual.Circle(win=win,radius=radius,
                         fillColor=(-1,-1,-1),lineWidth=lineWidth,
                         edges=edges,lineColor=(-1,-1,-1))
    crcl.draw()
    return crcl

def draw_fixation(win, x, y):
    """Function to draw fixation point"""
    fixation = visual.GratingStim(win=win,size=0.1,
                                  pos=[x,y],sf=0,
                                  color=(1,-1,-1))
    fixation.draw()
