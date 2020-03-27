import time
import random
import numpy as np
import pandas as pd
import configparser
from psychopy import visual, event, core, logging, gui, sound, monitors

parser = configparser.ConfigParser()
parser.read('config.ini')

# EXPERIMENTAL SETTINGS
TRIALS_PER_SPEED = int(parser['ExperimentOptions']['TrialsPerSpeed'])
FIXATOR_SIZE_PX = int(parser['ExperimentOptions']['FixatorSizePx'])

INIT_NO_ADAPTION_TIME = float(parser['ExperimentOptions']['InitialNoAdaptionTime'])
INIT_ADAPTION_TIME = float(parser['ExperimentOptions']['InitialAdaptionTime'])
POST_NO_ADAPTION_TIME = float(parser['ExperimentOptions']['PostNoAdaptionTime'])
POST_ADAPTION_TIME = float(parser['ExperimentOptions']['PostAdaptionTime'])
TEST_STIMULUS_TIME = float(parser['ExperimentOptions']['TestStimulusTime'])
ISI_TIME = float(parser['ExperimentOptions']['ISITime'])

# GRAPHICAL OPTIONS
BG_COLOR = "#636363"

# right = CCW, left = CW
key_list = ['right', 'left']
data_columns = ['Response', 'Time']

# Global variables
window = None
fixator = None
prompt = None

def main():
    global parser, window, fixator, prompt
    key_list.append('escape')

    # Prompt GUI for experimental setup
    dlg = gui.Dlg(title='Experiment Setup')
    dlg.addField('Participant ID:')                                                         # res[0]
    dlg.addField('Participant Age:')                                                        # res[1]
    dlg.addField('Participant Gender:', choices=['M', 'F'])                                 # res[2]
    dlg.addField('Experimenter ID:')                                                        # res[3]
    dlg.addField('Include Adaption:', 'Both', choices=['Both', 'No', 'Yes'])                # res[4]
    dlg.addField('Adaption Direction:', 'Random', choices=['Random', 'Clockwise', 'Counterclockwise'])      # res[5]
    res = dlg.show()
    if not dlg.OK:
        exit()

    dlg = gui.Dlg(title='Monitor Setup')
    dlg.addText('Monitor Info:')
    dlg.addField('Viewing distance (cm):', 57)
    dlg.addField('Width (cm):')
    dlg.addField('Width (pixels):')
    dlg.addField('Height (pixels):')
    monitor_info = dlg.show()
    if not dlg.OK:
        exit()
    viewing_dist, width_cm, width_px, height_px = [int(n) for n in monitor_info]

    exp_bg_info = res[:4]
    to_run_no_adaption = True if res[4] != 'Yes' else False
    to_run_adaption = True if res[4] != 'No' else False
    adaption_dir = res[5]

    #TODO: Gamma correction
    monitor = monitors.Monitor('monitor')
    monitor.setSizePix((height_px, width_px))
    monitor.setWidth(width_cm)
    monitor.setDistance(viewing_dist)
    window = visual.Window(color=BG_COLOR, monitor=monitor, fullscr=True) 

    refresh_rate = window.getActualFrameRate()   
    fixator = visual.Circle(window, size=(FIXATOR_SIZE_PX, FIXATOR_SIZE_PX), units='pix')

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
    prompt.text = 'The test will be a rotating grating. Press the left arrow key if it appears to be moving clockwise and the right arrow key if it appears to be moving counterclockwise.\n\nAfter the grating, you''ll see a blank screen again for 5 seconds, and then a beep and test image.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    grating_size = int(parser['GratingOptions']['GratingSize'])
    stim = visual.RadialStim(window, tex='sin', size=(grating_size, grating_size), units='deg', angularCycles=int(parser['GratingOptions']['GratingCycles']))

    # Create and loop through trials
    trials = []
    min_speed = float(parser['NoAdaptionOptions']['MinTestSpeed'])
    max_speed = float(parser['NoAdaptionOptions']['MaxTestSpeed'])
    step_size = float(parser['NoAdaptionOptions']['StepSize'])
    for _ in range(TRIALS_PER_SPEED):
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
    prompt = visual.TextStim(window, color='white', text='You will first see a rotating grating for 2 minutes. Try to keep your eyes fixed on the grating during this time.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()
    prompt.text = 'The test will be a rotating grating. Press the left arrow key if it appears to be moving clockwise and the right arrow key if it appears to be moving counterclockwise.\n\nAfter the grating, you''ll see a rotating grating again for 5 seconds, and then a beep and test grating.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    grating_size = int(parser['GratingOptions']['GratingSize'])
    stim = visual.RadialStim(window, tex='sin', size=(grating_size, grating_size), units='deg', angularCycles=int(parser['GratingOptions']['GratingCycles']))

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
    for _ in range(TRIALS_PER_SPEED):
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
