from psychopy import core, event, monitors, visual
from utils.stimulus import generate_stimulus

window = None

monitor = monitors.Monitor('monitor')
monitor.setSizePix((600, 600))
monitor.setWidth(30)
monitor.setDistance(54)

window = visual.Window(size=(600, 600), color="#A2A2A2", monitor=monitor, fullscr=False)
fixator = visual.Circle(window, size=(5, 5), color='red', units='pix')

stim_1 = generate_stimulus(window, transparent=False)
stim_1_t = generate_stimulus(window)
stim_2 = generate_stimulus(window, transparent=False, dir=-1)
stim_2_t = generate_stimulus(window, dir=-1)

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
