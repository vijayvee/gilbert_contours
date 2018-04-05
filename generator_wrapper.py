import gilbert_contours
import time
import sys
import numpy as np
import os
import gilbert_contours

class Args:
    def __init__(self,
                 contour_path = '', batch_id=0,
                 circle=False, curr_radius=4, zero_ecc=False, just_display=False, window_size=[256,256],
                 contour_length=15, n_images = 200000, shear_val=0, color_path = '',
                 contrast_range=[0.7, 1.], dist_uniform=False, scale_contrast=0.0,
                 random_contrast=False, random_contrast_std=-1, pause_display=False, distractor_contrast=1.0,
                 global_spacing = 0.25, save_images=False, randomContour=False, paddle_length=1.0,
                 color = False, random_rotations=False, skew_slack=2, zigzag=False, zigzagAngle=0):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.circle = circle
        self.curr_radius = curr_radius
        self.zero_ecc = zero_ecc
        self.just_display = just_display
        self.window_size = window_size
        self.contour_length = contour_length
        self.n_images = n_images
        self.shear_val = shear_val
        self.color_path = color_path
        self.contrast_range = contrast_range
        self.dist_uniform = dist_uniform
        self.scale_contrast = scale_contrast
        self.random_contrast = random_contrast
        self.random_contrast_std = random_contrast_std
        self.pause_display = pause_display
        self.distractor_contrast = distractor_contrast
        self.global_spacing = global_spacing
        self.save_images = save_images
        self.randomContour = randomContour
        self.paddle_length = paddle_length
        self.color =color
        self.random_rotations = random_rotations
        self.skew_slack = skew_slack
        self.zigzag = zigzag
        self.zigzagAngle = zigzagAngle

t = time.time()
args = Args()

## CONSTANTS
num_machines = int(sys.argv[1])
args.batch_id = int(sys.argv[2])
total_images = int(sys.argv[3])
args.n_images = total_images/num_machines

dataset_root = './' #'/gpfs/data/tserre/data/gilbert_contours/'

args.contour_length = 15
args.window_size = [128,128] #TODO: FIX
args.paddle_length = 0.1
args.random_rotations = True
args.skew_slack = 6.5

################################# DS: BASELINE
dataset_subpath = 'baseline'
args.contour_path = os.path.join(dataset_root, dataset_subpath)
gilbert_contours.from_wrapper(args)

################################# DS: uniform distractor contrast from some -1 to 1
for dc in [-1.0, -0.5, 0.0, 0.5]:
    dataset_subpath = 'dist_contrast_' + str(dc)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.distractor_contrast = dc
    gilbert_contours.from_wrapper(args)
args.distractor_contrast = 1.0

################################# DS: all paddle contrast with a distribution sigma from -1 to 0.7
args.random_contrast = True
#TODO: FIX
#for pc in [??]:
    #?????
    #contrast_range
    #dist_uniform
    #scale_contrast
    #random_contrast
    #random_contrast_std = -1
args.random_contrast = False

################################# DS: snake inter-paddle angle from 0 to 20
# NOT IMPLEMENTED YET
#TODO: FIX

################################# DS: snake inter-paddle distance from -0.7 to 0.7
for sv in [-0.7, -0.35, 0.35, 0.7]:
    dataset_subpath = 'shear_val_' + str(sv)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.shear_val = sv
    gilbert_contours.from_wrapper(args)
args.shear_val = 0

elapsed = time.time() - t

print('n_totl_imgs (per condition) : ', str(total_images))
print('ELAPSED TIME : ', str(elapsed))
