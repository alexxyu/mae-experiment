import math
import os

import matplotlib.pyplot as plt
import numpy as np
from psychopy import core, event, monitors, visual

window = None

def generate_stimulus(transparent = True, dir = 1):
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
        fig, ax = plt.subplots(facecolor='#A2A2A2')
        circle = plt.Circle((0, 0), radius, edgecolor=None, facecolor='#A2A2A2', fill=True)

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

        ax.plot(x_n, y_n, color=c, linewidth=5, linestyle='--')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.axis('off')

    if transparent:
        plt.savefig('stim.png', bbox_inches='tight', transparent = True)
    else:
        ax.add_artist(circle)
        plt.savefig('stim.png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')

    stim = visual.ImageStim(window, image='stim.png', units='deg', size=(10, 10), pos=(0, 0))
    os.remove('stim.png')
    return stim

monitor = monitors.Monitor('monitor')
monitor.setSizePix((600, 600))
monitor.setWidth(30)
monitor.setDistance(54)

window = visual.Window(size=(600, 600), color="#A2A2A2", monitor=monitor, fullscr=False)
fixator = visual.Circle(window, size=(5, 5), color='red', units='pix')

stim_1 = generate_stimulus(transparent=False)
stim_1_t = generate_stimulus()
stim_2 = generate_stimulus(transparent=False, dir=-1)
stim_2_t = generate_stimulus(dir=-1)

stim_1.draw()
stim_2_t.draw()
fixator.draw()
window.flip()

while len(event.getKeys()) == 0:
    stim_1.ori += 2
    stim_2_t.ori -= 2

    stim_1.draw()
    stim_2_t.draw()
    fixator.draw()
    window.flip()
    core.wait(0.02)
