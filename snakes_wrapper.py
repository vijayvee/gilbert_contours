import time
import sys
import numpy as np
import os
import snakes


class Args:
    def __init__(self,
                 contour_path = './contour', batch_id=0, n_images = 200000,
                 window_size=[256,256], antialias_scale = 4,
                 contour_length=15, distractor_length=5, num_distractor_contours=6, use_single_paddles=400,
                 max_target_contour_retrial = 4, max_distractor_contour_retrial = 4, max_paddle_retrial=2,
                 continuity = 1.4, paddle_length=5, paddle_thickness=1.5, paddle_margin_list=[4], paddle_contrast_list=[1.],
                 pause_display=False, save_images=True, save_gt=False, save_metadata=True):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.antialias_scale = antialias_scale

        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.num_distractor_contours = num_distractor_contours
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
        self.save_gt = save_gt
        self.save_metadata = save_metadata

t = time.time()
args = Args()
## Constraints
num_machines = int(sys.argv[1])
# start_id = int(sys.argv[2])
args.batch_id = int(sys.argv[2])  # start_id + int(sys.argv[3]) - 1
total_images = int(sys.argv[3])
args.n_images = total_images/num_machines

dataset_root = sys.argv[4]  #'/gpfs/data/tserre/data/PSST/curvy_decoy'


args.contour_length = 12 # from 6~18
args.antialias_scale = 2
args.continuity = 1.6  # from 2.4~0.8
args.distractor_length = args.contour_length / 3
args.num_distractor_contours = int(33 * (9. / args.contour_length))
args.use_single_paddles = False

################################# DS: snake length
# length from 10 to 30
for cl in [18, 21]:
    dataset_subpath = 'curv_contour_length_' + str(cl)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.contour_length = cl
    args.distractor_length = cl/3
    args.num_distractor_contours = int(30*(9./args.contour_length))
    args.LABEL = 1
    snakes.from_wrapper(args)
    dataset_subpath = 'curv_contour_length_' + str(cl) + '_neg'
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    args.contour_length = cl/3
    args.num_distractor_contours += 3
    args.LABEL = 0
    snakes.from_wrapper(args)

elapsed = time.time() - t
print('n_totl_imgs (per condition) : ', str(total_images))
print('ELAPSED TIME OVER ALL CONDITIONS : ', str(elapsed))
