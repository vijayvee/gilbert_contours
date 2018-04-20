import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
import scipy
from scipy import ndimage
import time
import cv2
cv2.useOptimized()

import snakes


def test():
    t = time.time()

    target_paddle_length = 25 # from 8 to 25
    distractor_paddle_length = target_paddle_length / 2
    num_distractor_paddles = 6 #4
    continuity = 1.5 #1  # from 1 to 2.5 (expect occasional failures at high values)

    imsize = 256
    aa_scale = 4
    segment_length = 5
    thickness = 1.5
    contrast_list = [1.0]
    margin = 4

    image = np.zeros((imsize, imsize))
    mask = np.zeros((imsize, imsize))

    ### TODO: missampled seed + % occupied constraint

    num_segments = target_paddle_length
    num_snakes = 1
    max_snake_trial = 10
    max_segment_trial = 2
    image1, mask = make_many_snakes(image, mask,
                                    num_snakes, max_snake_trial,
                                    num_segments, segment_length, thickness, margin, continuity, contrast_list,
                                    max_segment_trial, aa_scale,
                                    display_final=False, display_snake=False, display_segment=False,
                                    allow_incomplete=False, allow_shorter_snakes=False)
    num_segments = distractor_paddle_length
    num_snakes = num_distractor_paddles
    max_snake_trial = 4
    max_segment_trial = 2
    image2, mask = make_many_snakes(image1, mask,
                                    num_snakes, max_snake_trial,
                                    num_segments, segment_length, thickness, margin, continuity, contrast_list,
                                    max_segment_trial, aa_scale,
                                    display_final=False, display_snake=False, display_segment=False,
                                    allow_incomplete=False, allow_shorter_snakes=False)
    num_segments = 1
    num_snakes = 0 #400 - target_paddle_length - num_distractor_paddles * distractor_paddle_length
    max_snake_trial = 3
    max_segment_trial = 2
    image3, _ = make_many_snakes(image2, mask,
                                 num_snakes, max_snake_trial,
                                 num_segments, segment_length, thickness, margin, continuity, contrast_list,
                                 max_segment_trial, aa_scale,
                                 display_final=False, display_snake=False, display_segment=False,
                                 allow_incomplete=True, allow_shorter_snakes=False, stop_with_availability=0.01)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    red_target = gray2red(image1)
    show1 = scipy.misc.imresize(red_target, (imsize, imsize), interp='lanczos')
    plt.imshow(show1)
    plt.axis('off')

    plt.subplot(2, 1, 2)
    gray_total = gray2gray(1 - image3)
    show2 = scipy.misc.imresize(gray_total, (imsize, imsize), interp='lanczos')
    plt.imshow(show2)
    plt.axis('off')

    elapsed = time.time() - t
    print('ELAPSED TIME : ', str(elapsed))

    plt.show()

def from_wrapper(args):

    t = time.time()
    iimg = 0

    if (args.save_images):
        contour_sub_path = os.path.join('imgs', str(args.batch_id))
        if not os.path.exists(os.path.join(args.contour_path, contour_sub_path)):
            os.makedirs(os.path.join(args.contour_path, contour_sub_path))
    if (args.save_gt):
        gt_sub_path = os.path.join('gt_imgs', str(args.batch_id))
        if not os.path.exists(os.path.join(args.contour_path, gt_sub_path)):
            os.makedirs(os.path.join(args.contour_path, gt_sub_path))
    if args.save_metadata:
        metadata = []
        # CHECK IF METADATA FILE ALREADY EXISTS
        metadata_path = os.path.join(args.contour_path, 'metadata')
        if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
        metadata_fn = str(args.batch_id) + '.npy'
        metadata_full = os.path.join(metadata_path, metadata_fn)
        if os.path.exists(metadata_full):
            print('Metadata file already exists.')
            return

    while (iimg < args.n_images):
        print('Image# : %s'%(iimg))

        # Sample paddle margin
        num_possible_margins = len(args.paddle_margin_list)
        if num_possible_margins > 0:
            margin_index = np.random.randint(low=0, high=num_possible_margins)
        else:
            margin_index = 0
        margin = args.paddle_margin_list[margin_index]
        base_num_paddles = 400
        num_paddles_factor = 1./((7.5 + 13*margin + 4*margin*margin)/123.5)
        total_num_paddles = int(base_num_paddles*num_paddles_factor)

        image = np.zeros((args.window_size[0], args.window_size[1]))
        mask = np.zeros((args.window_size[0], args.window_size[1]))

        ############################################

        if (args.pause_display):
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            red_target = snakes.gray2red(1 - target_im)
            plt.imshow(red_target)
            plt.axis('off')
            plt.subplot(2, 1, 2)
            gray_total = snakes.gray2gray(final_im)
            plt.imshow(gray_total)
            plt.axis('off')
            plt.show()

        if (args.save_images):
            fn = "sample_%s.png"%(iimg)
            scipy.misc.imsave(os.path.join(args.contour_path, contour_sub_path, fn), final_im)
        if (args.save_gt):
            fn = "gt_%s.png"%(iimg)
            scipy.misc.imsave(os.path.join(args.contour_path, gt_sub_path, fn), target_im)
        if (args.save_metadata):
            metadata = snakes.accumulate_meta(metadata, contour_sub_path, fn, args, iimg, paddle_margin=margin)
            ## TODO: GT IS NOT INCLUDED IN METADATA
        iimg += 1

    if (args.save_metadata):
        matadata_nparray = np.array(metadata)
        snakes.save_metadata(matadata_nparray, args.contour_path, args.batch_id)
    elapsed = time.time() - t
    print('ELAPSED TIME : ', str(elapsed))

    plt.show()

    return

if __name__ == "__main__":
    test()

    # ALGORITHM
    # 1. compute initial point
    #    current_start = translate(last_endpoint, last_orientation, dilation+1)
    # 2. draw current_endpoint (distance = line_length + dilation)
    #    compute current_orientation
    #    M' <--- dilate(M, dilation+2)
    #    sample endpoint using M'
    #    trial_count += 1
    # 3. compute line and mask
    #    l_current, m_current = draw_line_n_mask(translate(current_start, current_orientation, dilation), current_endpoint, dilation)
    # 4. check if max(M + m_current) > 2
    #       yes -> check if retrial_count > max_count
    #           yes -> return with failure flag
    #           no -> goto 2
    #       no -> goto 5
    # 5. draw image I += l_current
    # 6. draw mask M = max(M, m_last)
    # 7. m_last = m_current.copy()
    # 8. retrial_count = 0
