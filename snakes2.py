import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import matplotlib.pyplot as plt
import PIL

from PIL import Image
from PIL import ImageDraw
import scipy
from scipy import ndimage
import time
import cv2
cv2.useOptimized()

import snakes

def imresize(arr, size):
    norm01 = False
    if int(arr.max()) == 1:
        old_dtype = arr.dtype
        norm01 = True
        arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize(size, PIL.Image.LANCZOS)
    npimg = np.array(img)
    if norm01:
        npimg = (npimg / 255).astype(old_dtype)
    return npimg

# Accumulate metadata
def accumulate_meta(array, subpath, filename, args, nimg, paddle_margin = None):
    # NEW VERSION
    array += [[subpath, filename, nimg, args.LABEL,
               args.continuity, args.contour_length, args.distractor_length,
               args.paddle_length, args.paddle_thickness, paddle_margin, len(args.paddle_contrast_list)]]
    return array
    # GENERATED ARRAY IS NATURALLY SORTED BY THE ORDER IN WHICH IMGS ARE CREATED.
    # IN TRAIN OR TEST TIME CALL np.random.shuffle(ARRAY)


def two_snakes(image_size, padding, seed_distance,
                    num_segments, segment_length, thickness, margin, continuity, small_dilation_structs, large_dilation_structs,
                    contrast_list,
                    max_segment_trial, aa_scale,
                    display_snake = False, display_segment = False,
                    allow_shorter_snakes=False, stop_with_availability=None):

    # draw initial segment
    for isegment in range(1):
        num_possible_contrasts = len(contrast_list)
        if num_possible_contrasts>0:
            contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
        else:
            contrast_index = 0
        contrast = contrast_list[contrast_index]
        current_image, current_mask, current_segment_masks, current_pivots, current_orientations, origin_tips, success \
        = initialize_two_seeds(image_size, padding, seed_distance,
                               segment_length, thickness, margin, contrast,
                               small_dilation_structs, large_dilation_structs,
                               max_segment_trial,
                               aa_scale, display=display_segment)
        if success is False:
            return np.zeros((image_size[0], image_size[1])), np.zeros((image_size[0], image_size[1])), None, None, False

    # sequentially add segments
    terminal_tips = [[0,0],[0,0]]
    for isegment in range(num_segments-1):
        if num_possible_contrasts>0:
            contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
        else:
            contrast_index = 0
        contrast = contrast_list[contrast_index]
        for isnake in range(len(current_segment_masks)):
            current_image, current_mask, current_segment_masks[isnake], current_pivots[isnake], current_orientations[isnake], terminal_tips[isnake], success \
            = snakes.extend_snake(list(current_pivots[isnake]), current_orientations[isnake], current_segment_masks[isnake],
                                  current_image, current_mask, max_segment_trial,
                                  segment_length, thickness, margin, continuity, contrast,
                                  small_dilation_structs, large_dilation_structs,
                                  aa_scale = aa_scale,
                                  display=display_segment,
                                  forced_current_pivot=None)
            if success is False:
                if allow_shorter_snakes:
                    return current_image, current_mask, None, None, True
                else:
                    return current_image, current_mask, None, None, False
    current_mask = np.maximum(current_mask, current_segment_masks[-1])
    # display snake
    if display_snake:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(current_image)
        plt.subplot(1, 2, 2)
        plt.imshow(current_mask)
        plt.show()
    return current_image, current_mask, origin_tips, terminal_tips, True


def initialize_two_seeds(image_size, padding, seed_distance,
                         length, thickness, margin, contrast,
                         small_dilation_structs, large_dilation_structs,
                         max_segment_trial,
                         aa_scale, display=False):

    image = np.zeros((image_size[0], image_size[1]))
    mask = np.zeros((image_size[0], image_size[1]))
    mask[:padding, :] = 1
    mask[-padding:, :] = 1
    mask[:, :padding] = 1
    mask[:, -padding:] = 1

    struct_shape = ((length+margin)*2+1, (length+margin)*2+1)
    struct_head = [length+margin+1, length+margin+1]

    ######################## SAMPLE FIRST SEGMENT
    trial_count = 0
    while trial_count <= max_segment_trial:
        sampled_orientation_in_rad1 = np.random.randint(low=-180, high=180) * np.pi / 180
        if sampled_orientation_in_rad1+np.pi < np.pi:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad1 + np.pi
        else:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad1 - np.pi

        # generate dilation struct
        _, struct = snakes.draw_line_n_mask(struct_shape, struct_head, sampled_orientation_in_rad1, length, thickness, margin, large_dilation_structs, aa_scale)
            # head-centric struct

        # dilate mask using segment
        lined_mask = mask.copy()
        lined_mask[:seed_distance*2,:] = 1
        lined_mask[image_size[0]-seed_distance*2:,:] = 1
        lined_mask[:,:seed_distance*2] = 1
        lined_mask[:,image_size[1]-seed_distance*2:] = 1
        dilated_mask = snakes.binary_dilate_custom(lined_mask, struct, value_scale=1.)
            # dilation in the same orientation as the tail

        # run coordinate searcher while also further dilating
        _, raw_num_available_coordinates = snakes.find_available_coordinates(np.ceil(mask-0.3), margin=0)
        available_coordinates, num_available_coordinates = snakes.find_available_coordinates(np.ceil(dilated_mask-0.3), margin=0)
        if num_available_coordinates == 0:
            #print('Mask fully occupied after dilation. finalizing')
            return image, mask, [np.zeros_like(mask),np.zeros_like(mask)], [None, None], [None, None], [None, None], False
            continue

        # sample coordinate and draw
        random_number = np.random.randint(low=0,high=num_available_coordinates)
        sampled_tail1 = [available_coordinates[0][random_number],available_coordinates[1][random_number]] # CHECK OUT OF BOUNDARY CASES
        sampled_head1 = snakes.translate_coord(sampled_tail1, sampled_orientation_in_rad1, length)
        sampled_pivot1 = snakes.translate_coord(sampled_head1, sampled_orientation_in_rad_reversed, length+margin)
        sampled_tip1 = [sampled_tail1[0], sampled_tail1[1]]
        if (sampled_head1[0] < 0) | (sampled_head1[0] >= mask.shape[0]) | \
           (sampled_head1[1] < 0) | (sampled_head1[1] >= mask.shape[1]) | \
           (sampled_pivot1[0] < 0) | (sampled_pivot1[0] >= mask.shape[0]) | \
           (sampled_pivot1[1] < 0) | (sampled_pivot1[1] >= mask.shape[1]):
            #print('missampled seed +segment_trial_count')
            trial_count += 1
            continue
        else:
            break
    if trial_count > max_segment_trial:
        return image, mask, [np.zeros_like(mask),np.zeros_like(mask)], [None, None], [None, None], [None, None], False
    l_im, m_im1 = snakes.draw_line_n_mask((mask.shape[0], mask.shape[1]), sampled_tail1, sampled_orientation_in_rad1, length, thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)
    image = np.maximum(image, l_im)


    ######################## SAMPLE SECOND SEGMENT
    trial_count = 0
    while trial_count <= max_segment_trial:
        sampled_orientation_in_rad2 = np.random.randint(low=-180, high=180) * np.pi / 180
        if sampled_orientation_in_rad2 + np.pi < np.pi:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad2 + np.pi
        else:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad2 - np.pi

        sample_in_rad = np.random.randint(0, 360) * np.pi / 180
        # get lists of y and x coordinates (exclude out-of-bound coordinates)
        sample_in_y = int(np.round_(sampled_tail1[0] + (seed_distance * np.sin(sample_in_rad))))
        sample_in_x = int(np.round_(sampled_tail1[1] + (seed_distance * np.cos(sample_in_rad))))
        sampled_tail2 = [sample_in_y, sample_in_x]
        sampled_head2 = snakes.translate_coord(sampled_tail2, sampled_orientation_in_rad2, length)
        sampled_pivot2 = snakes.translate_coord(sampled_head2, sampled_orientation_in_rad_reversed, length + margin)
        sampled_tip2 = [sampled_tail2[0], sampled_tail2[1]]
        if (sampled_head2[0] < 0) | (sampled_head2[0] >= mask.shape[0]) | \
           (sampled_head2[1] < 0) | (sampled_head2[1] >= mask.shape[1]) | \
           (sampled_pivot2[0] < 0) | (sampled_pivot2[0] >= mask.shape[0]) | \
           (sampled_pivot2[1] < 0) | (sampled_pivot2[1] >= mask.shape[1]):
            #print('missampled seed +segment_trial_count')
            trial_count += 1
            continue
        else:
            break
    if trial_count > max_segment_trial:
        return image, mask, [np.zeros_like(mask),np.zeros_like(mask)], [None, None], [None, None], [None, None], False

    l_im, m_im2 = snakes.draw_line_n_mask((mask.shape[0], mask.shape[1]), sampled_tail2, sampled_orientation_in_rad2, length,
                                  thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)
    image = np.maximum(image, l_im)

    if display:
        plt.figure(figsize=(10,20))
        plt.imshow(image)
        plt.title(str(num_available_coordinates))
        plt.plot(sampled_tail1[1], sampled_tail1[0], 'bo')
        plt.plot(sampled_head1[1], sampled_head1[0], 'ro')
        plt.plot(sampled_tail2[1], sampled_tail2[0], 'bo')
        plt.plot(sampled_head2[1], sampled_head2[0], 'ro')
        plt.show()

    return image, mask, [m_im1,m_im2], [sampled_pivot1, sampled_pivot2], [sampled_orientation_in_rad1, sampled_orientation_in_rad2], [sampled_tip1, sampled_tip2], True


def draw_circle(window_size, coordinate, radius, aa_scale):
    image = np.zeros((window_size[0]*aa_scale, window_size[1]*aa_scale))
    y, x = np.ogrid[-coordinate[0]*aa_scale:(window_size[0]-coordinate[0])*aa_scale,
                    -coordinate[1]*aa_scale:(window_size[1]-coordinate[1])*aa_scale]
    mask = x ** 2 + y ** 2 <= (radius*aa_scale) ** 2
    # image[mask] = 1
    image[mask] = 255
    # return scipy.misc.imresize(image, (window_size[0], window_size[1]), interp='lanczos')
    return imresize(image, (window_size[0], window_size[1]))

def test():
    t = time.time()

    imsize = 300
    padding = 22
    aa_scale = 4

    LABEL = 1

    snake_length = 18 # from 5 to 21, with steps of 4
    distractor_length = snake_length/3
    continuity = 2.5  # from 3.0 ~ 1.0 with steps of -0.5
    base_num_paddles = 150

    segment_length = 5
    marker_radius = 3
    thickness = 1.5
    margin = 4
    seed_distance = 3*(segment_length+margin)

    contrast_list = [1.0]
    max_distractor_snake_retrial = 3
    max_segment_trial = 2
    num_paddles_factor = 1. / ((7.5 + 13 * margin + 4 * margin * margin) / 123.5)
    total_num_paddles = int(base_num_paddles * num_paddles_factor)

    ### SAMPLE TWO TARGET SNAKES
    small_dilation_structs = snakes.generate_dilation_struct(margin)
    large_dilation_structs = snakes.generate_dilation_struct(margin*aa_scale)
    success = False
    while not success:
        image, mask, origin_tips, terminal_tips, success = \
        two_snakes([imsize, imsize], padding, seed_distance,
                   snake_length, segment_length, thickness, margin, continuity,
                   small_dilation_structs, large_dilation_structs,
                   contrast_list,
                   max_segment_trial, aa_scale,
                   display_snake=False, display_segment=False,
                   allow_shorter_snakes=False, stop_with_availability=None)

    ### SAMPLE SHORT SNAKE DISTRACTORS
    num_distractor_snakes = 40/distractor_length
    image, mask = snakes.make_many_snakes(image, mask,
                                       num_distractor_snakes, max_distractor_snake_retrial,
                                       distractor_length, segment_length, thickness, margin, continuity, contrast_list,
                                        max_segment_trial, aa_scale,
                                        display_final=False, display_snake=False, display_segment=False,
                                        allow_incomplete=True, allow_shorter_snakes=False, stop_with_availability=0.01)


    ### SAMPLE SINGLE PADDLE DISTRACTORS
    num_single_paddles = total_num_paddles - 2*snake_length - num_distractor_snakes*distractor_length
    image, _ = snakes.make_many_snakes(image, mask,
                                        num_single_paddles, max_segment_trial,
                                       1, segment_length, thickness, margin, continuity, contrast_list,
                                        max_segment_trial, aa_scale,
                                        display_final=False, display_snake=False, display_segment=False,
                                        allow_incomplete=True, allow_shorter_snakes=False, stop_with_availability=0.01)

    ### ADD MARKERS
    if LABEL == 0:
        origin_mark_idx = np.random.randint(0,2)
        terminal_mark_idx = 1-origin_mark_idx
    else:
        origin_mark_idx = np.random.randint(0,2)
        terminal_mark_idx = origin_mark_idx
    origin_mark_coord = origin_tips[origin_mark_idx]
    terminal_mark_coord = terminal_tips[terminal_mark_idx]
    markers = np.maximum(draw_circle([imsize,imsize], origin_mark_coord, marker_radius, aa_scale),
                        draw_circle([imsize,imsize], terminal_mark_coord, marker_radius, aa_scale)).astype(np.float)/255
    image_marked = np.maximum(image, markers)

    elapsed = time.time() - t

    plt.figure(figsize=(10, 10))
    # show2 = scipy.misc.imresize(image_marked, (imsize, imsize), interp='lanczos')
    show2 = imresize(image_marked, (imsize, imsize))
    plt.imshow(show2)
    plt.colorbar()
    plt.axis('off')
    plt.show()

    print('ELAPSED TIME : ', str(elapsed))



def from_wrapper(args):

    t = time.time()
    iimg = 0

    if (args.save_images):
        contour_sub_path = os.path.join('imgs', str(args.batch_id))
        if not os.path.exists(os.path.join(args.contour_path, contour_sub_path)):
            os.makedirs(os.path.join(args.contour_path, contour_sub_path))
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
        base_num_paddles = 150
        num_paddles_factor = 1. / ((7.5 + 13 * margin + 4 * margin * margin) / 123.5)
        total_num_paddles = int(base_num_paddles * num_paddles_factor)

        small_dilation_structs = snakes.generate_dilation_struct(margin)
        large_dilation_structs = snakes.generate_dilation_struct(margin * args.antialias_scale)

        ### SAMPLE TWO TARGET SNAKES
        success = False
        while not success:
            image, mask, origin_tips, terminal_tips, success = \
                two_snakes(args.window_size, args.padding, args.seed_distance,
                           args.contour_length, args.paddle_length, args.paddle_thickness, margin, args.continuity,
                           small_dilation_structs, large_dilation_structs,
                           args.paddle_contrast_list,
                           args.max_paddle_retrial,
                           args.antialias_scale,
                           display_snake=False, display_segment=False,
                           allow_shorter_snakes=False, stop_with_availability=None)

        ### SAMPLE SHORT SNAKE DISTRACTORS
        num_distractor_snakes = 30 / args.distractor_length
        image, mask = snakes.make_many_snakes(image, mask,
                                              num_distractor_snakes, args.max_distractor_contour_retrial,
                                              args.distractor_length, args.paddle_length, args.paddle_thickness, margin, args.continuity,
                                              args.paddle_contrast_list,
                                              args.max_paddle_retrial,
                                              args.antialias_scale,
                                              display_final=False, display_snake=False, display_segment=False,
                                              allow_incomplete=True, allow_shorter_snakes=False,
                                              stop_with_availability=0.01)

        if (image is None):
            continue
        if args.use_single_paddles is not False:
            ### SAMPLE SINGLE PADDLE DISTRACTORS
            num_single_paddles = total_num_paddles - 2 * args.contour_length - num_distractor_snakes * args.distractor_length
            image, _ = snakes.make_many_snakes(image, mask,
                                               num_single_paddles, args.max_paddle_retrial,
                                               1, args.paddle_length, args.paddle_thickness, margin, args.continuity,
                                               args.paddle_contrast_list,
                                               args.max_paddle_retrial,
                                               args.antialias_scale,
                                               display_final=False, display_snake=False, display_segment=False,
                                               allow_incomplete=True, allow_shorter_snakes=False,
                                               stop_with_availability=0.01)
            if (image is None):
                continue

        ### ADD MARKERS
        if args.LABEL == 0:
            origin_mark_idx = np.random.randint(0, 2)
            terminal_mark_idx = 1 - origin_mark_idx
        else:
            origin_mark_idx = np.random.randint(0, 2)
            terminal_mark_idx = origin_mark_idx
        origin_mark_coord = origin_tips[origin_mark_idx]
        terminal_mark_coord = terminal_tips[terminal_mark_idx]
        markers = np.maximum(draw_circle(args.window_size, origin_mark_coord, args.marker_radius, args.antialias_scale),
                             draw_circle(args.window_size, terminal_mark_coord, args.marker_radius, args.antialias_scale)).astype(np.float) / 255
        image_marked = np.maximum(image, markers)

        if (args.pause_display):
            plt.figure(figsize=(10, 10))
            # show2 = scipy.misc.imresize(image_marked, (args.window_size[0], args.window_size[1]), interp='lanczos')
            show2 = imresize(image_marked, (args.window_size[0], args.window_size[1]))
            plt.imshow(show2)
            plt.colorbar()
            plt.axis('off')
            plt.show()
        if (args.save_images):
            fn = "sample_%s.png"%(iimg)
            # scipy.misc.imsave(os.path.join(args.contour_path, contour_sub_path, fn), image_marked)
            print(image_marked.min(), image_marked.max(), image_marked.dtype)
            image_marked = np.clip(image_marked, 0, 1)
            image_marked = (image_marked * 255).astype(np.uint8)
            print(image_marked.min(), image_marked.max(), image_marked.dtype)
            pil_image_marked = Image.fromarray(image_marked)
            pil_image_marked.save(os.path.join(args.contour_path, contour_sub_path, fn))
            del pil_image_marked
        if (args.save_metadata):
            metadata = accumulate_meta(metadata, contour_sub_path, fn, args, iimg, paddle_margin=margin)
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
