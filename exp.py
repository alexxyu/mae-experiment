import os
import time
import random
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt
from psychopy import visual, event, core, gui, sound, monitors

random.seed(time.time())
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
data_columns = ['Response', 'Time', 'Test Speed']
experiment_info_columns = ['Subject', 'Age', 'Gender', 'Experimenter', 'Seq No', 'Adapting', 'Adapt Dir', 'View Dist (cm)', 'Width (cm)', 'Width (px)', 'Height (px)', 'Refresh Rate (Hz)']

# Global variables
window = None
fixator = None
prompt = None
subject = None
seqNo = None

def main():
    global parser, window, fixator, prompt, subject, seqNo
    key_list.append('escape')

    # Prompt GUI for experimental setup
    dlg = gui.Dlg(title='Experiment Setup')
    dlg.addField('Participant ID:')                                                         # res[0]
    dlg.addField('Participant Age:')                                                        # res[1]
    dlg.addField('Participant Gender:', choices=['M', 'F'])                                 # res[2]
    dlg.addField('Experimenter ID:')                                                        # res[3]
    dlg.addField('Sequence Number:')                                                        # res[4]
    dlg.addField('Include Adaption:', 'Both', choices=['Both', 'No', 'Yes'])                # res[5]
    dlg.addField('Adaption Direction:', 'Random', choices=['Random', 'Clockwise', 'Counterclockwise'])      # res[6]
    exp_res = dlg.show()
    if not dlg.OK:
        exit()

    if ((exp_res[5] != 'Yes' and os.path.exists(f'data/{exp_res[0]}{exp_res[4]}_noAdapt.csv')) or 
       (exp_res[5] != 'No' and os.path.exists(f'data/{exp_res[0]}{exp_res[4]}_Adapt.csv'))):
       warning = gui.Dlg(title='Duplicate Filename Warning')
       warning.addText('An existing file with the same name has been found. Do you want to overwrite it?')
       res = warning.show()
       if not warning.OK:
           exit()

    # Prompt GUI for monitor setup
    dlg = gui.Dlg(title='Monitor Setup')
    dlg.addText('Monitor Info:')
    dlg.addField('Viewing distance (cm):', '57')
    dlg.addField('Width (cm):')
    dlg.addField('Width (pixels):')
    dlg.addField('Height (pixels):')

    monitor_info = []
    while True:
        monitor_info = dlg.show()
        if not dlg.OK:
            exit()
        if '' not in monitor_info and all([s.isdigit() for s in monitor_info]):
            break
        else:
            dlg = gui.Dlg(title='Monitor Setup')
            dlg.addText('Please fill out all fields and enter integer values only.')
            dlg.addText('Monitor Info:')
            dlg.addField('Viewing distance (cm):', '57')
            dlg.addField('Width (cm):')
            dlg.addField('Width (pixels):')
            dlg.addField('Height (pixels):')

    viewing_dist, width_cm, width_px, height_px = [int(n) for n in monitor_info]

    subject = exp_res[0]
    seqNo = exp_res[4]
    to_run_no_adaption = True if exp_res[5] != 'Yes' else False
    to_run_adaption = True if exp_res[5] != 'No' else False
    adaption_dir = exp_res[6]

    # Create monitor profile and window for experiment
    monitor = monitors.Monitor('monitor')
    monitor.setSizePix((height_px, width_px))
    monitor.setWidth(width_cm)
    monitor.setDistance(viewing_dist)
    window = visual.Window(color=BG_COLOR, monitor=monitor, fullscr=True) 
    window.mouseVisible = False

    gamma_table = np.load('gammaTable.npy')
    gamma_ramp = np.vstack((gamma_table, gamma_table, gamma_table))
    window.gammaRamp = gamma_ramp

    refresh_rate = round(window.getActualFrameRate())
    fixator = visual.Circle(window, size=(FIXATOR_SIZE_PX, FIXATOR_SIZE_PX), units='pix')

    # Save experiment setup data to file
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/experiment_info.csv'):
        df = pd.DataFrame(columns=experiment_info_columns)
    else:
        df = pd.read_csv('data/experiment_info.csv', index_col=0)
    df.loc[len(df)] = exp_res + monitor_info + [refresh_rate]
    df.to_csv('data/experiment_info.csv')

    # Run experiment parts
    if to_run_no_adaption:
        run_without_adaption(window)
    if to_run_adaption:
        if to_run_no_adaption and to_run_adaption:
            prompt.text = 'You have finished the first part of the experiment. We will now move on to the second part.\n\nPress any key to continue.'
            prompt.draw()
            window.flip()
            event.waitKeys()
        run_with_adaption(window, adaption_dir)

    quit(True)

def run_without_adaption(window):
    global parser, prompt, subject
    data = pd.DataFrame(columns=data_columns)

    # Experimental instructions
    prompt = visual.TextStim(window, color='white', text=f'You will see a blank screen for {round(INIT_NO_ADAPTION_TIME)} seconds. You do not have to press anything. Try to keep your eyes fixed during this time.\n\nThen, you will hear a beep to alert you that a test is about to begin.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()
    prompt.text = f'The test will be a rotating grating. Press the right arrow key if it appears to be rotating to the right (clockwise) and the left arrow key if it appears to be rotating to the left (counterclockwise).\n\nAfter the grating, you\'\'ll see a blank screen again for {round(POST_NO_ADAPTION_TIME)} seconds, and then a beep and test image.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
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
        # Top-up adaptor (blank screen)
        fixator.color = 'red'
        fixator.draw()
        window.flip()
        if is_first_trial:
            wait_time = INIT_NO_ADAPTION_TIME
            is_first_trial = False
        else:
            wait_time = POST_NO_ADAPTION_TIME
        start_time = time.time()
        while time.time() - start_time < wait_time:
            if len(event.getKeys(keyList=['escape'])) > 0:
                quit()

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

        # Wait for response
        window.flip()
        if len(res) == 0:
            res = event.waitKeys()
        if 'escape' in res:
            quit()

        end_time = time.time()
        data.loc[len(data)] = [res[0], end_time - start_time, trial_speed]
        data.to_csv(f'data/{subject}{seqNo}_noAdapt.csv')
        save_psychometric_plot(data, 'No', 'noAdapt')

def run_with_adaption(window, adaption_dir):
    global parser, prompt, subject
    data = pd.DataFrame(columns=data_columns)

    # Experimental instructions
    prompt = visual.TextStim(window, color='white', text=f'You will first see a rotating grating for {round(INIT_ADAPTION_TIME)} seconds. Try to keep your eyes fixed on the grating during this time.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()
    prompt.text = f'The test will be a rotating grating. Press the right arrow key if it appears to be rotating to the right (clockwise) and the left arrow key if it appears to be rotating to the left (counterclockwise).\n\nAfter the grating, you\'\'ll see a rotating grating again for {round(POST_ADAPTION_TIME)} seconds, and then a beep and test grating.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    grating_size = int(parser['GratingOptions']['GratingSize'])
    stim = visual.RadialStim(window, tex='sin', size=(grating_size, grating_size), units='deg', angularCycles=int(parser['GratingOptions']['GratingCycles']))

    # Generate adaptor speeds
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
        data.loc[len(data)] = [res[0], end_time - start_time, trial_speed]
        data.to_csv(f'data/{subject}{seqNo}_Adapt.csv')
        save_psychometric_plot(data, adaption_dir, 'Adapt')

def isi(window):
    fixator.draw()
    window.flip()
    core.wait(ISI_TIME/2)
    beep = sound.Sound('A', secs=0.2)
    beep.play()
    fixator.color = 'green'
    fixator.draw()
    window.flip()
    core.wait(ISI_TIME/2)

def quit(finished = False):
    if finished:
        prompt.text = 'You have completed the study.\n\nPress any key to exit.'
    else:
        prompt.text = 'The experiment has been stopped.\n\nPress any key to exit.'
    prompt.draw()
    window.flip()
    event.waitKeys()
    window.close()
    exit()

def save_psychometric_plot(data, adapt, adaptName):
    count_data = data.groupby('Test Speed')['Response'].apply(lambda x: x[x.str.contains('right')].count())
    total_counts = data.groupby('Test Speed').count()['Response']
    speeds = count_data.index.tolist()
    right_counts = count_data.tolist()
    right_props = [r / c for r, c in zip(right_counts, total_counts)]

    plt.clf()
    plt.plot(speeds, right_props)
    plt.axhline(y=0.5, color='r', linewidth=0.5, linestyle='dashed')
    plt.axvline(x=0, color='r', linewidth=0.5, linestyle='dashed')
    plt.title(f'Subject {subject}{seqNo} ({adapt} Adapt)')
    plt.xlabel('Test Stimulus Speed (deg/s)')
    plt.ylabel('Proportion of Responding Clockwise')
    plt.savefig(f'data/{subject}{seqNo}_{adaptName}_plot.png')

if __name__ == "__main__":
    main()
