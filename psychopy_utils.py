#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
from tqdm import tqdm
import os
from random import sample
from matplotlib import pyplot as plt

def create_window(window_size, monitor):
    #create a window
    mywin = visual.Window([window_size[0], window_size[1]] ,color=(-1,-1,-1), monitor=monitor, units="deg")
    return mywin

def draw_line(win, start, end, lineWidth=3.):
    #function to draw a line
    ln = visual.Line(win=win,start=start,end=end,lineWidth=lineWidth,lineColor=(1,-1,-1))
    ln.draw()

def draw_grating_stim(win, pos, size,ori=0, lineColor=(1,1,1)):
    stim = visual.GratingStim(win=win, pos=pos, size=(0.3,1.0), tex='sin', mask='cross', units='deg', sf=1.5, ori=ori, color=lineColor)
    return stim

def draw_circle(win,radius,lineWidth=3.,edges=48):
    crcl = visual.Circle(win=win,radius=radius,fillColor=(-1,-1,-1),lineWidth=lineWidth,edges=edges,lineColor=(-1,-1,-1))
    crcl.draw()
    return crcl

def draw_fixation(win, x, y):
    fixation = visual.GratingStim(win=win, size=0.1, pos=[x,y], sf=0, color=(1,-1,-1))
    fixation.draw()
