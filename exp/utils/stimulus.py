import math
import os

import matplotlib.pyplot as plt
import numpy as np
from psychopy import visual


def generate_stimulus(window, transparent = True, size = 10, dir = 1, linewidth = 5, linestyle='--', n_phase = 4, bg_color='#A2A2A2'):
    b = 0.7
    n_phase = 4

    t = np.linspace(0, 2*np.pi, 100)
    phi_vals = np.linspace(0, 2*np.pi, 1000)
    phase_thresh = np.pi / n_phase
    phase_number = 1
    radius = np.exp(2*np.pi*b)

    if transparent:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(facecolor=bg_color)
        circle = plt.Circle((0, 0), radius, edgecolor=None, facecolor=bg_color, fill=True)

    for phi in phi_vals:
        c = 'white'
        if phi > phase_number * phase_thresh:
            c = 'black'
            phase_number += 1
        elif phi == 0:
            c = 'black'
        else:
            continue

        x = np.cos(dir*(t - phi)) * np.exp(b*t)
        y = np.sin(dir*(t - phi)) * np.exp(b*t)

        x_n = [x for x, y in zip(x, y) if math.sqrt(x**2 + y**2) >= 20]
        y_n = [y for x, y in zip(x, y) if math.sqrt(x**2 + y**2) >= 20]

        ax.plot(x_n, y_n, color=c, linewidth=linewidth, linestyle=linestyle)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.axis('off')

    if transparent:
        plt.savefig('stim.png', bbox_inches='tight', transparent = True)
    else:
        ax.add_artist(circle)
        plt.savefig('stim.png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')

    stim = visual.ImageStim(window, image='stim.png', units='deg', size=(size, size), pos=(0, 0))
    os.remove('stim.png')
    return stim
