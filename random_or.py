import random

import psychopy.visual
import psychopy.event

win = psychopy.visual.Window(
    size=[256, 256],
    units="pix",
    fullscr=False, monitor='testMonitor'
)

n_dots = 200

dot_xys = []
dot_oris = []

for dot in range(n_dots):

    dot_x = random.uniform(-8, 8)
    dot_y = random.uniform(-8, 8)

    dot_xys.append([dot_x, dot_y])

    dot_oris.append(random.uniform(0, 180))
print dot_xys
dot_stim = psychopy.visual.ElementArrayStim(
    win=win,
    units="deg",
    nElements=n_dots,
    elementTex="sin",
    elementMask="gauss",
    sfs=5.0 / 2.5,
    xys=dot_xys,
    sizes=0.7,
    oris=dot_oris
)

dot_stim.draw()

win.flip()

psychopy.event.waitKeys()

win.close()
