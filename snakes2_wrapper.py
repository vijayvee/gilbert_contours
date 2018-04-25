import time
import sys
import numpy as np
import os
import snakes2

class Args:
    def __init__(self,
                 contour_path = './contour', batch_id=0, n_images = 200000,
                 window_size=[256,256], padding=22, antialias_scale = 4,
                 LABEL =1, seed_distance= 27, marker_radius = 3,
                 contour_length=15, distractor_length=5, use_single_paddles=True,
                 max_target_contour_retrial = 4, max_distractor_contour_retrial = 4, max_paddle_retrial=2,
                 continuity = 1.4, paddle_length=5, paddle_thickness=1.5, paddle_margin_list=[4], paddle_contrast_list=[1.],
                 pause_display=False, save_images=True, save_metadata=True):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.padding = padding
        self.antialias_scale = antialias_scale

        self.LABEL = LABEL
        self.seed_distance = seed_distance
        self.marker_radius = marker_radius
        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.use_single_paddles = use_single_paddles

        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial

        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list # if multiple elements in a list, a number will be sampled in each IMAGE
        self.paddle_contrast_list = paddle_contrast_list # if multiple elements in a list, a number will be sampled in each PADDLE

        self.pause_display = pause_display
        self.save_images = save_images
        self.save_metadata = save_metadata

t = time.time()
args = Args()
## Constraints
num_machines = int(sys.argv[1])
start_id = int(sys.argv[2])
args.batch_id = start_id + int(sys.argv[3]) - 1
total_images = int(sys.argv[4])
args.n_images = total_images/num_machines

dataset_root = '/media/data_cifs/curvy_2snakes/'
args.antialias_scale = 4

args.marker_radius = 3
args.contour_length = 9 # from 9 to 18, with steps of 3
args.antialias_scale = 2
args.continuity = 1.8  # from 2.7, 1.8, 1.2, 0.8, 0.6
args.distractor_length = args.contour_length / 3
args.use_single_paddles = False

################################# DS: BASELINE
dataset_subpath = 'curv_baseline'
args.contour_path = os.path.join(dataset_root, dataset_subpath)
args.LABEL = 1
#snakes2.from_wrapper(args)
dataset_subpath = 'curv_baseline_neg'
args.contour_path = os.path.join(dataset_root, dataset_subpath)
args.LABEL = 0
#snakes2.from_wrapper(args)

################################# DS: snake length
for cl in [18]: #[6, 18]:
    args.contour_length = cl
    args.distractor_length = cl/3
    dataset_subpath = 'curv_contour_length_' + str(cl)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.LABEL = 1
    snakes2.from_wrapper(args)
    dataset_subpath = 'curv_contour_length_' + str(cl) + '_neg'
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.LABEL = 0
    snakes2.from_wrapper(args)
args.contour_length = 9
args.distractor_length = 3

################################# DS: snake inter-paddle continuity
for ct in [0.6]: #[2.7, 0.6]:
    args.continuity = ct
    dataset_subpath = 'curv_continuity_' + str(ct)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.LABEL = 1
    snakes2.from_wrapper(args)
    dataset_subpath = 'curv_continuity_' + str(ct) + '_neg'
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.LABEL = 0
    snakes2.from_wrapper(args)
args.continuity = 2.5

################################# DS: (REST OF THE 2-way MATRIX)
# NOT IMPLEMENTED YET
for cl in [6, 12, 15, 18]:
    for ct in [2.7, 1.2, 0.8, 0.6]:
        if (cl==6) & (ct ==2.7):
            args.continuity = ct
            args.contour_length = cl
            args.distractor_length = cl / 3
            # POS
            dataset_subpath = 'curv_continuity_' + str(ct) + '_length_' + str(cl)
            args.contour_path = os.path.join(dataset_root, dataset_subpath)
            args.LABEL = 1
            # snakes2.from_wrapper(args)
            # NEG
            dataset_subpath = 'curv_continuity_' + str(ct) + '_length_' + str(cl) + '_neg'
            args.contour_path = os.path.join(dataset_root, dataset_subpath)
            args.LABEL = 0
            # snakes2.from_wrapper(args)
        else:
            print('not implemented.')


elapsed = time.time() - t
print('n_totl_imgs (per condition) : ', str(total_images))
print('ELAPSED TIME OVER ALL CONDITIONS : ', str(elapsed))
