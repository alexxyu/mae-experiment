import time
import random
import pandas as pd
from psychopy import visual, event, core, logging, gui, sound

#TODO: Add speed, radius, and num cycles as free parameters
grating_cycles = 12

# GRAPHICAL OPTIONS
bg_color = "#827F7B"
fixator_dims = (5, 5)
beep = sound.Sound('A', secs=0.2)

# right = CCW, left = CW
key_list = ['right', 'left']
data_columns = ['Response', 'Time']

def main():

    # Prompt GUI for experimental setup
    dlg = gui.Dlg(title='Experiment Setup')
    dlg.addText('Participant Info:')
    dlg.addField('ID:')
    dlg.addField('Age:')
    dlg.addField('Gender:', choices=['M', 'F'])
    dlg.addText('Experiment Setup:')
    dlg.addField('Experimenter ID:')
    dlg.addField('Include Adaption:', 'Both', choices=['Both', 'No', 'Yes'])
    dlg.addField('Adaption Direction:', 'Random', choices=['Random', 'Left', 'Right'])
    res = dlg.show()
    if not dlg.OK:
        print('Program aborted.')
        exit()
    exp_bg_info = res[:4]

    data = pd.DataFrame(columns=data_columns)

    #TODO: Gamma correction
    #TODO: Display and stim size correction
    window = visual.Window(color=bg_color, monitor='monitor', fullscr=True)
    fixator = visual.Circle(window, size=fixator_dims, units='pix')

    #TODO: Prompts
    prompt = visual.TextStim(window, color='white', text='You will see a blank screen for 30 seconds. You do not have to press anything. Try to keep your eyes fixed during this time.\n\nThen, you will hear a beep to alert you that a test is about to begin.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()

    prompt.text = 'The test will be a rotating grating. Press the left arrow key if it appears to be moving clockwise and the right arrow key if it appears to be moving counterclockwise.\n\nAfter the grating, you''ll see a rotating grating again for 5 seconds, and then a beep and test image.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    # Loop through trials:
    stim = visual.RadialStim(window, tex='sin', size=(200, 200), units='pix', angularCycles=grating_cycles)

    # Initial blank screen
    fixator.color = 'red'
    fixator.draw()
    window.flip()
    core.wait(2)

    # ISI
    core.wait(0.25)
    beep.play()
    fixator.color = 'green'
    fixator.draw()
    window.flip()
    core.wait(0.25)

    res = []
    start_time = time.time()
    while len(res) == 0:
        res = event.getKeys(keyList=key_list)

        stim.draw()
        window.flip()
        stim.ori = (stim.ori + 2) % 360

        core.wait(0.02)

    stim.draw()
    window.flip()

    #TODO: handle experimental data export
    #end_time = time.time()
    #data.loc[len(data)] = [res[0], end_time - start_time]

    prompt.text = 'You have completed the study.\n\nPress any key to exit.'
    prompt.draw()
    window.flip()
    event.waitKeys()

if __name__ == "__main__":
    main()
