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

def main():
    contour_path = '.'
    make_contours_dir(contour_path)
    monitor = monitors.Monitor('SonyG55', distance=57)
    monitor.setSizePix([256,256])
    monitor.setWidth(30)
    win = create_window([256,256],monitor=monitor)
    draw_grating_stim(win,(0,0),0.05)
    import ipdb; ipdb.set_trace()
    for i in np.arange(0,1,0.25):
        for j in np.arange(0,1,0.25):
            print i,j
            draw_line(win, (i,j), True, True, size=1, ori=0-45, center=False)
            win.update()

    win.close()

if __name__=="__main__":
    main()
