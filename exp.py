import time
import random
import pandas as pd
from psychopy import visual, event, core, logging, gui, sound

# EXPERIMENTAL SETTINGS
#TODO: Add speed, radius, and num cycles as free parameters
grating_cycles = 8
degrees_per_sec = 360

init_no_adaption_time = 2
init_adaption_time = 2
post_no_adaption_time = 5
post_adaption_time = 5
test_stimulus_time = 10

# GRAPHICAL OPTIONS
bg_color = "#636363"

# right = CCW, left = CW
key_list = ['right', 'left']
data_columns = ['Response', 'Time']

# Global variables
window = None
fixator = None
prompt = None

def main():
    global window, fixator, prompt
    key_list.append('escape')

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

    to_run_no_adaption = True if res[5] != 'Yes' else False
    to_run_adaption = True if res[5] != 'No' else False

    #TODO: Gamma correction
    #TODO: Display and stim size correction
    window = visual.Window(size=(600,600), color=bg_color, monitor='monitor', fullscr=False)    
    fixator = visual.Circle(window, size=(5, 5), units='pix')

    if to_run_no_adaption:
        run_without_adaption(window)
    if to_run_adaption:
        run_with_adaption(window)

    #TODO: handle experimental data export

    prompt.text = 'You have completed the study.\n\nPress any key to exit.'
    prompt.draw()
    window.flip()
    event.waitKeys()
    window.close()

def quit():
    prompt.text = 'The experiment has been stopped.\n\nPress any key to exit.'
    prompt.draw()
    window.flip()
    window.close()
    exit()

def run_without_adaption(window):
    global prompt
    data = pd.DataFrame(columns=data_columns)

    # Experimental instructions
    prompt = visual.TextStim(window, color='white', text='You will see a blank screen for 30 seconds. You do not have to press anything. Try to keep your eyes fixed during this time.\n\nThen, you will hear a beep to alert you that a test is about to begin.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()
    prompt.text = 'The test will be a rotating grating. Press the left arrow key if it appears to be moving clockwise and the right arrow key if it appears to be moving counterclockwise.\n\nAfter the grating, you''ll see a rotating grating again for 5 seconds, and then a beep and test image.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    #TODO: Stimulus sizing
    stim = visual.RadialStim(window, tex='sin', size=(200, 200), units='pix', angularCycles=grating_cycles)

    # Loop through trials
    is_first_trial = True
    for _ in range(5):
        # Top-up adaptor
        fixator.color = 'red'
        fixator.draw()
        window.flip()
        if is_first_trial:
            core.wait(init_no_adaption_time)
            is_first_trial = False
        else:
            core.wait(post_no_adaption_time)

        # ISI
        isi(window)

        # Test stimulus
        res = []
        start_time = time.time()
        while len(res) == 0 and time.time() - start_time < test_stimulus_time:
            res = event.getKeys(keyList=key_list)
            if 'escape' in res:
                quit()

            stim.draw()
            fixator.draw()
            window.flip()
            stim.ori = (stim.ori + degrees_per_sec/100) % 360

            core.wait(0.01)

        window.flip()
        if len(res) == 0:
            res = event.waitKeys()
        if 'escape' in res:
            quit()

        end_time = time.time()
        data.loc[len(data)] = [res[0], end_time - start_time]

    #TODO: handle experimental data export
    #data.to_csv('')

def run_with_adaption(window):
    global prompt
    data = pd.DataFrame(columns=data_columns)

    # Experimental instructions
    prompt = visual.TextStim(window, color='white', text='You will see a blank screen for 30 seconds. You do not have to press anything. Try to keep your eyes fixed during this time.\n\nThen, you will hear a beep to alert you that a test is about to begin.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()
    prompt.text = 'The test will be a rotating grating. Press the left arrow key if it appears to be moving clockwise and the right arrow key if it appears to be moving counterclockwise.\n\nAfter the grating, you''ll see a rotating grating again for 5 seconds, and then a beep and test image.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    #TODO: Stimulus sizing
    stim = visual.RadialStim(window, tex='sin', size=(200, 200), units='pix', angularCycles=grating_cycles)

    # Loop through trials
    is_first_trial = True
    for _ in range(2):
        # Top up adaptor
        fixator.color = 'red'
        adaption_time = init_adaption_time if is_first_trial else post_adaption_time
        if is_first_trial:
            is_first_trial = False
        while len(res) == 0 and time.time() - start_time < adaption_time:
            res = event.getKeys(keyList=key_list)
            if 'escape' in res:
                quit()

            stim.draw()
            fixator.draw()
            window.flip()
            stim.ori = (stim.ori + degrees_per_sec/100) % 360

            core.wait(0.01)

        # ISI
        isi(window)

        # Test stimulus
        res = []
        start_time = time.time()
        while len(res) == 0 and time.time() - start_time < test_stimulus_time:
            res = event.getKeys(keyList=key_list)
            if 'escape' in res:
                quit()

            stim.draw()
            window.flip()
            stim.ori = (stim.ori + degrees_per_sec/100) % 360

            core.wait(0.01)

        window.flip()
        if len(res) == 0:
            res = event.waitKeys()
        if 'escape' in res:
            quit()

        end_time = time.time()
        data.loc[len(data)] = [res[0], end_time - start_time]

    #TODO: handle experimental data export
    #data.to_csv('')

def isi(window):
    window.flip()
    core.wait(0.25)
    beep = sound.Sound('A', secs=0.2)
    beep.play()
    fixator.color = 'green'
    fixator.draw()
    window.flip()
    core.wait(0.25)

if __name__ == "__main__":
    main()
