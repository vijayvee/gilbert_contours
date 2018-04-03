import gilbert_contours
import time
import sys
import numpy as np
import os
import gilbert_contours

def Args(contour_path = '', batch_id=0,
        circle=False, curr_radius=4, zero_ecc=False, just_display=False, window_size=[400,400],
        contour_lengths=[15], n_images = 200000, shear_range=[-0.7,0.7], color_path = '',
        contrast_range=[0.7, 1.], dist_uniform=False, scale_contrast=0.0,
        random_contrast=False, random_contrast_std=0.5, pause_display=False, distractor_contrast=[1.0],
        global_spacing = 0.25, save_images=False, randomContour=False, paddle_length=1.0,
        color = False, random_rotations=False, skew_slack=2, zigzag=False, zigzagAngle=0):

    Args.contour_path = contour_path
    Args.batch_id = batch_id
    Args.circle = circle
    Args.curr_radius = curr_radius
    Args.zero_ecc = zero_ecc
    Args.just_display = just_display
    Args.window_size = window_size
    Args.contour_lengths = contour_lengths
    Args.n_images = n_images
    Args.shear_range = shear_range
    Args.color_path = color_path
    Args.contrast_range = contrast_range
    Args.dist_uniform = dist_uniform
    Args.scale_contrast = scale_contrast
    Args.random_contrast = random_contrast
    Args.random_contrast_std = random_contrast_std
    Args.pause_display = pause_display
    Args.distractor_contrast = distractor_contrast
    Args.global_spacing = global_spacing
    Args.save_images = save_images
    Args.randomContour = randomContour
    Args.paddle_length = paddle_length
    Args.color =color
    Args.random_rotations = random_rotations
    Args.skew_slack = skew_slack
    Args.zigzag = zigzag
    Args.zigzagAngle = zigzagAngle

t = time.time()
args = Args()

## CONSTANTS
num_machines = int(sys.argv[1])
i_machine = int(sys.argv[2])

dataset_root = '/gpfs/data/tserre/data/gilbert_contours/'
args.radius = 4
args.window_size = [256,256]
args.lengths = [15]
args.n_images = 30
args.paddle_length = 0.1
args.random_rotations = True
args.skew_slack = 6.5

batches = num_machines

## EXPERIMENTAL VARIABLES
# DREW:
# uniform distractor contrast from some -1 to 1
# all paddle contrast with a distribution sigma from -1 to 0.7
# snake inter-paddle angle from 0 to 20
# snake inter-paddle distance from -0.7 to 0.7
args.lengths = [5,9,11]

kth_job = 0

for iparam, param in params:
    dataset_subpath = str(param)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)

    for ibatch in range(batches):

        kth_job += 1
        if np.mod(kth_job, num_machines) != i_machine and num_machines != i_machine:
            continue
        elif np.mod(kth_job, num_machines) != 0 and num_machines == i_machine:
            continue

        args.batch_id = ibatch

        gilbert_contours.from_wrapper(args)

                

elapsed = time.time() - t

print('ELAPSED TIME : ', str(elapsed))

