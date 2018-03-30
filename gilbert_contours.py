#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
import scipy.stats as stats
import argparse
from tqdm import tqdm
import os
from random import sample
from matplotlib import pyplot as plt
from gilbert_utils import *
from geometric_transformations import *
from psychopy_utils import *

"""Code to create the snakes dataset for association fields"""

CURR_RADIUS = 4
CONTOUR_PATH = '.'
WINDOW_SIZE = [400,400]
CONTOUR_LENGTHS = [15]
N_IMAGES = 200000
SHEAR_RANGE = [-0.7,0.7]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dir', dest='contour_path',
            default=CONTOUR_PATH, help='Directory where contour images are stored')
    parser.add_argument(
            '--circle', dest='circle', type=bool,
            default=False, help='Retain circle? (True/False)')
    parser.add_argument(
            '--radius', dest='curr_radius', type=int,
            default=CURR_RADIUS, help='Radius of circle for gilbert stimuli')
    parser.add_argument(
            '--zero_ecc', dest='zero_ecc', type=bool,
    default=False, help='Radius of circle for gilbert stimuli')
    parser.add_argument(
            '--just_display', dest='just_display', type=bool,
            default=False, help='Just display don\'t save?(True/False)')
    parser.add_argument(
            '--window_size', dest='window_size', nargs='+', type=int,
            default=WINDOW_SIZE, help='Size of stimulus window')
    parser.add_argument(
            '--lengths', dest='contour_lengths', nargs='+', type=int,
            default=CONTOUR_LENGTHS, help='Lengths of contour snakes')
    parser.add_argument(
            '--n_images', dest='n_images', type=int,
            default=N_IMAGES, help='Number of contour images to render')
    parser.add_argument(
            '--shear_range', dest='shear_range', nargs='+', type=float,
            default=SHEAR_RANGE, help='Range of shear for stimuli')
    parser.add_argument(
            '--color_path', dest='color_path', type=str,
            default='', help='Path to store color labels')
    parser.add_argument(
            '--random_contrast', dest='random_contrast', type=bool,
            default=False, help='Stable contrast for stimuli? (True/False)')
    parser.add_argument(
            '--random_contrast_std', dest='random_contrast_std', type=float,
            default=0.5, help='Stable contrast for stimuli? (True/False)')
    parser.add_argument(
            '--pause_display', dest='pause_display', type=bool,
            default=False, help='Pause display after rendering contour? (True/False)')
    parser.add_argument(
            '--distractor_contrast', dest='distractor_contrast', nargs='+', type=float,
            default=[1.0], help='Contrast value for distractor? (0-1)')
    parser.add_argument(
            '--global_spacing', dest='global_spacing', type=float,
            default=0.25, help='Space between any two snakes')
    parser.add_argument(
            '--save_images', dest='save_images', type=bool,
            default=False, help='Save images?')
    parser.add_argument(
            '--random_contour', dest='randomContour', type=bool,
            default=False, help='Random orientations for contour? (True/False)')
    parser.add_argument(
            '--paddle_length', dest='paddle_length', type=float,
            default=0.1, help='Length of the paddle forming snakes')
    parser.add_argument(
            '--color', dest='color', type=bool,
            default=False, help='Print contours in color? (True/False)')
    parser.add_argument(
            '--random_rotations', dest='rotate', type=bool,
            default=False, help='Rotate window randomly? (True/False)')
    parser.add_argument(
            '--skew_slack', dest='skew_slack', type=float,
            default=2, help='Slack positions for compensating skew?')
    parser.add_argument(
            '--zigzag', dest='zigzag', type=bool,
            default=False, help='Zigzag contours?')
    parser.add_argument(
            '--zigzag_angle', dest='zigzagAngle', type=float,
            default=0, help='Zigzag angle?')
    args = parser.parse_args()
    return args

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

def fillIncludeContour_zigzag(nRows,nCols, pos, length=5):
    #Include contours by relaxing the tangential contour path condition
    includeContour = np.zeros((nRows,nCols))
    if length == 0:
        return includeContour
    includeContour[pos[0],pos[1]] = 1
    bounds = int((length+1)/2)
    for ind,i in enumerate(np.arange(-bounds,bounds,1)):
        if ind%2==0:
            includeContour[pos[0]+i,pos[1]+i] = 1
        else:
            includeContour[pos[0]+i,pos[1]+i] = -1
    return includeContour


def draw_lines_row(win, win2, circle, positions, args,
                    color=False, size=0.1, shearAngle=0.3,
                    length=3, contourPosition=(10,20),
                    randomContrast=True, zigzag=False, zigzagAngle=0,
                    distractor_contrast=1.0):
    """Function to draw the main contour line segments."""
    #Set includeContour[i,j]=1. if position i,j of contour grid is along the contour path
    if zigzag:
        includeContour = fillIncludeContour_zigzag(nRows=positions.shape[0],
                                                     nCols=positions.shape[1],
                                                     pos=contourPosition,
                                                     length=length)
    else:
        includeContour = fillIncludeContour_nontan(nRows=positions.shape[0],
                                                     nCols=positions.shape[1],
                                                     pos=contourPosition,
                                                     length=length)
    positions_ = shear(positions,shearX=shearAngle) #Shearing to adjust intra-contour
    oriContour,contour,center = -1,False,False
    ori_orth = np.random.uniform(-180,180) #Choose random starting angle for distractor line segments
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            #Main loop, align elements along the contour and randomly orient other elements in contour grid
            pos = positions_[i,j,:]
            if circle is not None:
                if not circle.contains(pos, units='deg'):
                    continue
            #if circle.contains(pos,units='deg'):
            if np.abs(includeContour[i,j])==1:
                if args.randomContour:
                    ori = np.random.uniform(0,180)
                else:
                    ori = zigzagAngle*includeContour[i,j] + getContourOrientation(shearAngle)
                if not randomContrast:
                    contrast = 1.
                else:
                    contrast = np.random.uniform(-args.distractor_contrast, args.distractor_contrast)
                contour=True
            else:
                alpha, beta = np.abs(ori_orth+45), np.abs(ori_orth+90) #Driving neighbouring 'distractor' lines to be non-collinear
                minTheta, maxTheta = min(alpha,beta), max(alpha,beta) #Range of orientation of new 'distractor' line segment
                ori_orth = np.random.uniform(minTheta, maxTheta)
                if ori_orth<0:
                    ori_orth += 360 #If angle negative, add 360. <= a = 2.pi + a
                if not randomContrast:
                    contrast = distractor_contrast
                ori = ori_orth
                contour=False
            draw_line(win, pos=pos, contour=contour, color=color, size=size, ori=ori, center=center, contrast=contrast)
            if win2 is not None:
                draw_line(win2, pos=pos, contour=contour, color=True, size=size, ori=ori, center=center, contrast=contrast)

def main_drawing_loop(win, args, win2=None):
    #Main loop for rendering contours
    nLinesOnDiameter = len(np.arange(-args.skew_slack,
                                    args.skew_slack,
                                    args.global_spacing))
    nLinesOnRadius = len(np.arange(-args.skew_slack,
                                    0,
                                    args.global_spacing))
    circle = None
    if args.circle:
        circle = draw_circle(win=win, radius=args.curr_radius)
    if win2 is not None:
        if args.circle:
            draw_circle(win=win2, radius=args.curr_radius)
    linesPerDegree = deg2lines(radiusDegrees=args.curr_radius,
                                nLinesOnRadius=nLinesOnRadius)
    min_ecc, max_ecc = get_eccentricity_bounds(curr_radius=args.curr_radius,
                                               gilb_radius=43.8/2, gilb_min_ecc=2.4,
                                               gilb_max_ecc=8.4)
    max_ecc = min(2*min_ecc, max_ecc)
    mean_ecc = (min_ecc + max_ecc)/2.
    scale_ecc = max_ecc - min_ecc
    norm_ecc = stats.truncnorm((min_ecc - mean_ecc)/scale_ecc,
                                 (max_ecc-mean_ecc)/scale_ecc,
                                 loc=mean_ecc, scale=scale_ecc)
    print "Eccentricity bounds: %s %s"%(min_ecc ,max_ecc)
    shearLow, shearHigh = min(args.shear_range), max(args.shear_range)
    for shearAngle in [shearLow]:
        print "Shear only :", shearLow
        for contrast in args.distractor_contrast:
            print "Contrast only:", args.distractor_contrast
            for length in tqdm(args.contour_lengths,total=1, desc='Generating multiple length contours'):
                print "Length only:",args.contour_lengths
                for nimg in tqdm(range(args.n_images),desc='Generating contours for length %s'%(length)):
                    viewOri = np.random.uniform(0,360)
                    win.viewOri = viewOri
                    if win2 is not None:
                        win2.viewOri = viewOri
                    #curr_ecc = np.random.uniform(min_ecc, max_ecc)
                    if not args.zero_ecc:
                        curr_ecc = norm_ecc.rvs()
                    else:
                        curr_ecc = 0
                    ecc_lines = curr_ecc*linesPerDegree
                    a_contour = np.random.uniform(0,np.pi/2)
                    pos = get_contour_center(a_contour, curr_ecc)
                    pos = nLinesOnRadius+int(linesPerDegree*(pos[1])), nLinesOnRadius-int(linesPerDegree*(pos[0]))
                    positions = [(j,i) for i in np.arange(-args.skew_slack,
                                                            args.skew_slack,
                                                            args.global_spacing)
                                       for j in np.arange(-args.skew_slack,
                                                            args.skew_slack,
                                                            args.global_spacing)]
                    print "Contour center: %s,%s"%(pos)
                    positions = np.array(positions).reshape((
                                                        nLinesOnDiameter,
                                                        nLinesOnDiameter,2))
                    draw_lines_row(win, win2, circle, positions, args,
                                    color=args.color,length=length,
                                    shearAngle=shearAngle,size=args.paddle_length,
                                    randomContrast=args.random_contrast,
                                    contourPosition=pos, zigzag=args.zigzag,
                                    zigzagAngle=args.zigzagAngle,
                                    distractor_contrast=contrast)
                    win.flip()
                    if win2 is not None:
                        win2.flip()
                    if args.pause_display:
                        import ipdb; ipdb.set_trace()
                    win.getMovieFrame()

                    if args.just_display:
                        continue
                    if args.randomContour:
                        win.saveMovieFrames("%s/sample_%s_contrast_%s_shear_%s_length0_eccentricity_%s_%s.png"
                                               %(args.contour_path,
                                                   contrast,
                                                   args.window_size[0],
                                                   shearAngle,
                                                   curr_ecc,nimg))
                        if win2 is not None:
                            win2.getMovieFrame()
                            win2.saveMovieFrames("%s/sample_color_%s_contrast_%s_shear_%s_length0_eccentricity_%s_%s.png"
                                                   %(args.color_path,
                                                       contrast,
                                                       args.window_size[0],
                                                       shearAngle,
                                                       curr_ecc,nimg))
                    else:
                        win.saveMovieFrames("%s/sample_%s_contrast_%s_shear_%s_length%s_eccentricity_%s_%s.png"
                                               %(args.contour_path,
                                                   contrast,
                                                   args.window_size[0],
                                                   shearAngle,length,
                                                   curr_ecc,nimg))
                        if win2 is not None:
                            win2.getMovieFrame()
                            win2.saveMovieFrames("%s/sample_color_%s_contrast_%s_shear_%s_length%s_eccentricity_%s_%s.png"
                                                   %(args.color_path,
                                                       contrast,
                                                       args.window_size[0],
                                                       shearAngle,length,
                                                       curr_ecc,nimg))
    win.close()


def main():
    args = parse_arguments()
    make_contours_dir(args.contour_path)
    win2 = None
    print args.window_size
    if args.color_path != '':
        make_contours_dir(args.color_path)
        win2 = create_window(args.window_size,monitor='testMonitor')
    win = create_window(args.window_size,monitor='testMonitor')
    print "Created window"
    main_drawing_loop(win, args, win2)

if __name__=="__main__":
    main()
