#!/usr/bin/python
from psychopy import visual, event, monitors, core
import numpy as np
import scipy.linalg
import scipy.stats as stats
import argparse
from tqdm import tqdm
from argparser import *
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
    return_positions, return_orientations = [], []
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
    if args.random_contrast:
        minContrast, maxContrast = args.contrast_range
        normContrast = get_normal_dist(minContrast, maxContrast)
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
                    if args.uniform_contrast:
                        contrast = np.random.uniform(minContrast, maxContrast)
                    else:
                        contrast = normContrast.rvs() #np.random.uniform(-args.distractor_contrast, args.distractor_contrast)
                contour=True
            else:
                alpha, beta = np.abs(ori_orth+45), np.abs(ori_orth+90) #Driving neighbouring 'distractor' lines to be non-collinear
                minTheta, maxTheta = min(alpha,beta), max(alpha,beta) #Range of orientation of new 'distractor' line segment
                ori_orth = np.random.uniform(minTheta, maxTheta)
                if ori_orth<0:
                    ori_orth += 360 #If angle negative, add 360. <= a = 2.pi + a
                if not randomContrast:
                    contrast = distractor_contrast
                else:
                    if args.uniform_contrast:
                        contrast = np.random.uniform(minContrast, maxContrast)
                    else:
                        contrast = normContrast.rvs() #np.random.uniform(low=0., high=1.0)
                ori = ori_orth
                contour=False
            return_positions.append(pos)
            return_orientations.append(ori)
            draw_line(win, pos=pos, contour=contour, color=color, size=size, ori=ori, center=center, contrast=contrast)
            if win2 is not None:
                draw_line(win2, pos=pos, contour=contour, color=True, size=size, ori=ori, center=center, contrast=contrast)
    return return_positions, return_orientations

def accumulate_meta(array, subpath, filename, randomContour, contrast, window_size, shearAngle, length, curr_ecc, nimg):
    if length is None:
        length = -1 # DEFAULT FOR NONE CASE
    array += [[subpath, filename, nimg, shearAngle, contrast, window_size, randomContour, length, curr_ecc]]
    return array
    # GENERATED ARRAY IS NATURALLY SORTED BY THE ORDER IN WHICH IMGS ARE CREATED.
    # IN TRAIN OR TEST TIME CALL np.random.shuffle(ARRAY)

def save_metadata(metadata, args):
    # Converts metadata (list of lists) into an nparray, and then saves
    metadata_path = os.path.join(args.contour_path, 'metadata')
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    metadata_fn = str(args.batch_id) + '.npy'
    np.save(os.path.join(metadata_path,metadata_fn), metadata)

def main_drawing_loop(win, args, win2=None, metadata=None): ###########
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
    norm_ecc = get_normal_dist(min_ecc, max_ecc)
    print "Eccentricity bounds: %s %s"%(min_ecc ,max_ecc)
    shearAngle = args.shear_val #min(args.shear_range), max(args.shear_range)
    shearMin, shearMax = shearAngle-args.global_spacing, shearAngle
    shearNorm = get_normal_dist(shearMin, shearMax)
    print "Random contrast: %s"%(args.random_contrast)
    contrast = args.distractor_contrast
    length = args.contour_length
    image_sub_path = os.path.join('imgs', str(args.batch_id))
    make_contours_dir(os.path.join(args.contour_path, image_sub_path))
    #make_contours_dir(os.path.join(args.color_path, image_sub_path))
    for nimg in tqdm(range(args.n_images),desc='Generating contours for length %s'%(length)):
        shearAngle = shearNorm.rvs()
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
            ###########
            filename = "sample_%s_contrast_%s_shear_%s_length0_eccentricity_%s_%s.png"\
                                   %(contrast,
                                       args.window_size[0],
                                       shearAngle,
                                       curr_ecc,nimg)
            win.saveMovieFrames(os.path.join(args.contour_path,image_sub_path,filename))
            if metadata is not None:
                metadata = accumulate_meta(metadata, image_sub_path, filename,
                                        args.randomContour, contrast, args.window_size[0], shearAngle, None, curr_ecc,
                                        nimg)
            ###########
            if win2 is not None:
                win2.getMovieFrame()
                filename = "sample_color_%s_contrast_%s_shear_%s_length0_eccentricity_%s_%s.png" \
                           % (contrast,
                              args.window_size[0],
                              shearAngle,
                              curr_ecc, nimg)
                win2.saveMovieFrames(os.path.join(args.color_path, image_sub_path, filename))
        else:
            ###########
            filename = "sample_%s_contrast_%s_shear_%s_length%s_eccentricity_%s_%s.png"\
                                   %(contrast,
                                       args.window_size[0],
                                       shearAngle, length,
                                       curr_ecc,nimg)
            win.saveMovieFrames(os.path.join(args.contour_path, image_sub_path, filename))
            if metadata is not None:
                metadata = accumulate_meta(metadata, image_sub_path, filename,
                                        args.randomContour, contrast, args.window_size[0], shearAngle, length, curr_ecc,
                                        nimg)
            ###########
            if win2 is not None:
                win2.getMovieFrame()
                filename = "sample_color_%s_contrast_%s_shear_%s_length%s_eccentricity_%s_%s.png" \
                           % (contrast,
                              args.window_size[0],
                              shearAngle, length,
                              curr_ecc, nimg)
                win2.saveMovieFrames(os.path.join(args.color_path, image_sub_path, filename))
    win.close()
    return metadata


def main():
    CURR_RADIUS = 4
    CONTOUR_PATH = '.'
    WINDOW_SIZE = [400, 400]
    contour_length = 15
    N_IMAGES = 200000
    SHEAR_VAL = -0.7

    args = parse_arguments()
    make_contours_dir(args.contour_path)
    win2 = None
    print args.window_size
    if args.color_path != '':
        make_contours_dir(args.color_path)
        win2 = create_window(args.window_size,monitor='testMonitor')
    win = create_window(args.window_size,monitor='testMonitor')
    print "Created window"
    metadata = main_drawing_loop(win, args, win2, metadata = [])
    matadata_nparray = np.array(metadata)
    save_metadata(matadata_nparray, args)

def from_wrapper(args):
    make_contours_dir(args.contour_path)
    win2 = None
    print args.window_size
    if args.color_path != '':
        make_contours_dir(args.color_path)
        win2 = create_window(args.window_size,monitor='testMonitor')
    win = create_window(args.window_size,monitor='testMonitor')
    print "Created window"
    metadata = main_drawing_loop(win, args, win2, metadata = [])
    matadata_nparray = np.array(metadata)
    save_metadata(matadata_nparray, args)

if __name__=="__main__":
    main()
    # python gilbert_contours.py --dir ./ --metadir ./ --radius 4 --window_size 256 256 --lengths 15 --n_images 200000 --shear_range -0.7 --global_spacing 0.25 --paddle_length 0.1 --random_rotations True --skew_slack 6.5
