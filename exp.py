import time
import random
import numpy as np
import pandas as pd
import configparser
from psychopy import visual, event, core, logging, gui, sound

# EXPERIMENTAL SETTINGS
trials_per_speed = 20

INIT_NO_ADAPTION_TIME = 30
INIT_ADAPTION_TIME = 120
POST_NO_ADAPTION_TIME = 5
POST_ADAPTION_TIME = 5
TEST_STIMULUS_TIME = 0.075
ISI_TIME = 0.5

# GRAPHICAL OPTIONS
BG_COLOR = "#636363"

# right = CCW, left = CW
key_list = ['right', 'left']
data_columns = ['Response', 'Time']

# Global variables
parser = configparser.ConfigParser()
parser.read('config.ini')
window = None
fixator = None
prompt = None

def main():
    global parser, window, fixator, prompt
    key_list.append('escape')

    # Prompt GUI for experimental setup
    dlg = gui.Dlg(title='Experiment Setup')
    dlg.addText('Participant Info:')
    dlg.addField('ID:')                                                                     # res[0]
    dlg.addField('Age:')                                                                    # res[1]
    dlg.addField('Gender:', choices=['M', 'F'])                                             # res[2]
    dlg.addText('Experiment Setup:')
    dlg.addField('Experimenter ID:')                                                        # res[3]
    dlg.addField('Include Adaption:', 'Both', choices=['Both', 'No', 'Yes'])                # res[4]
    dlg.addField('Adaption Direction:', 'Random', choices=['Random', 'Clockwise', 'Counterclockwise'])      # res[5]
    res = dlg.show()
    if not dlg.OK:
        exit()
    exp_bg_info = res[:4]

    to_run_no_adaption = True if res[4] != 'Yes' else False
    to_run_adaption = True if res[4] != 'No' else False
    adaption_dir = res[5]

    #TODO: Gamma correction
    window = visual.Window(size=(600,600), color=BG_COLOR, monitor='monitor', fullscr=False)    
    fixator = visual.Circle(window, size=(5, 5), units='pix')

    if to_run_no_adaption:
        run_without_adaption(window)
    if to_run_adaption:
        run_with_adaption(window, adaption_dir)

    #TODO: handle experimental data export

    prompt.text = 'You have completed the study.\n\nPress any key to exit.'
    prompt.draw()
    window.flip()
    event.waitKeys()
    window.close()

def run_without_adaption(window):
    global parser, prompt
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
    stim = visual.RadialStim(window, tex='sin', size=(200, 200), units='pix', angularCycles=int(parser['GratingOptions']['GratingCycles']))

    # Create and loop through trials
    trials = []
    min_speed = float(parser['NoAdaptionOptions']['MinTestSpeed'])
    max_speed = float(parser['NoAdaptionOptions']['MaxTestSpeed'])
    step_size = float(parser['NoAdaptionOptions']['StepSize'])
    for i in range(trials_per_speed):
        temp = [round(n, 5) for n in np.arange(min_speed, max_speed + step_size, step=step_size)]
        trials.extend(temp)
    random.shuffle(trials)

    is_first_trial = True
    for trial_speed in trials:
        # Top-up adaptor
        fixator.color = 'red'
        fixator.draw()
        window.flip()
        if is_first_trial:
            core.wait(INIT_NO_ADAPTION_TIME)
            is_first_trial = False
        else:
            core.wait(POST_NO_ADAPTION_TIME)

        # ISI
        isi(window)

        # Test stimulus
        res = []
        event.clearEvents()
        start_time = time.time()
        while len(res) == 0 and time.time() - start_time < TEST_STIMULUS_TIME:
            res = event.getKeys(keyList=key_list)
            if 'escape' in res:
                quit()

            stim.draw()
            fixator.draw()
            window.flip()
            stim.ori = stim.ori + trial_speed/100

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

def run_with_adaption(window, adaption_dir):
    global parser, prompt
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
    stim = visual.RadialStim(window, tex='sin', size=(200, 200), units='pix', angularCycles=int(parser['GratingOptions']['GratingCycles']))

    if adaption_dir == 'Random':
        adaption_dir = 'Clockwise' if random.getrandbits(1) == 0 else 'Counterclockwise'
    if adaption_dir == 'Clockwise':
        min_speed = float(parser['CWAdaptionOptions']['MinTestSpeed'])
        max_speed = float(parser['CWAdaptionOptions']['MaxTestSpeed'])
        step_size = float(parser['CWAdaptionOptions']['StepSize'])
    else:
        min_speed = float(parser['CCWAdaptionOptions']['MinTestSpeed'])
        max_speed = float(parser['CCWAdaptionOptions']['MaxTestSpeed'])
        step_size = float(parser['CCWAdaptionOptions']['StepSize'])

    trials = []
    for i in range(trials_per_speed):
        temp = [round(n, 5) for n in np.arange(min_speed, max_speed + step_size, step=step_size)]
        trials.extend(temp)
    random.shuffle(trials)

    # Loop through trials
    is_first_trial = True
    adaptor_speed = float(parser['GratingOptions']['AdaptorSpeed'])
    adaptor_speed = adaptor_speed if adaption_dir == 'Clockwise' else -adaptor_speed
    for trial_speed in trials:
        # Top up adaptor
        fixator.color = 'red'
        adaption_time = INIT_ADAPTION_TIME if is_first_trial else POST_ADAPTION_TIME
        if is_first_trial:
            is_first_trial = False
        start_time = time.time()
        while time.time() - start_time < adaption_time:
            if len(event.getKeys(keyList=['escape'])) > 0:
                quit()

            stim.draw()
            fixator.draw()
            window.flip()
            stim.ori = (stim.ori + adaptor_speed/100) % 360

            core.wait(0.01)

        # ISI
        isi(window)

        # Test stimulus
        res = []
        event.clearEvents()
        start_time = time.time()
        while len(res) == 0 and time.time() - start_time < TEST_STIMULUS_TIME:
            res = event.getKeys(keyList=key_list)
            if 'escape' in res:
                quit()

            stim.draw()
            fixator.draw()
            window.flip()
            stim.ori = (stim.ori + trial_speed/100) % 360

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
    core.wait(ISI_TIME/2)
    beep = sound.Sound('A', secs=0.2)
    beep.play()
    fixator.color = 'green'
    fixator.draw()
    window.flip()
    core.wait(ISI_TIME/2)

def quit():
    prompt.text = 'The experiment has been stopped.\n\nPress any key to exit.'
    prompt.draw()
    window.flip()
    window.close()
    exit()

if __name__ == "__main__":
    main()
