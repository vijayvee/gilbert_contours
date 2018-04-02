import argparse

CURR_RADIUS = 4
CONTOUR_PATH = '.'
WINDOW_SIZE = [400,400]
CONTOUR_LENGTHS = [15]
N_IMAGES = 200000
SHEAR_RANGE = [-0.7,0.7]

def parse_arguments():
    #Parser to parse command line arguments
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
            default=False, help='Plot contours only in the circle center? (True/False)')
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
            '--contrast_range', dest='contrast_range', type=float, nargs='+',
            default=[0.7,1.], help='Minimum and maximum contrast [-1,1]')
    parser.add_argument(
            '--uniform', dest='dist_uniform', type=bool,
            default=False, help='Random distributions uniform? (True - Uniform. False - Normal.)')
    parser.add_argument(
            '--scale_contrast', dest='scale_contrast', type=float,
            default=0.0, help='Standard deviation of contrast range')
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
            default=False, help='Render contours in color? (True/False)')
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
