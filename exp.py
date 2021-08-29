import os
import time
import math
import random
import numpy as np
import pandas as pd
import configparser
from datetime import datetime
import matplotlib.pyplot as plt
from psychopy import visual, event, core, gui, sound, monitors

random.seed(time.time())
parser = configparser.ConfigParser()
parser.read('config.ini')

# EXPERIMENTAL SETTINGS
STIM_SIZE = int(parser['StimulusOptions']['StimSize'])
NUM_PHASES = int(parser['StimulusOptions']['StimCycles'])

TRIALS_PER_COMBINATION = int(parser['ExperimentOptions']['TrialsPerCombo'])
FIXATOR_SIZE_PX = int(parser['ExperimentOptions']['FixatorSizePx'])

INIT_NO_ADAPTION_TIME = float(parser['ExperimentOptions']['InitialNoAdaptionTime'])
INIT_ADAPTION_TIME = float(parser['ExperimentOptions']['InitialAdaptionTime'])
POST_NO_ADAPTION_TIME = float(parser['ExperimentOptions']['PostNoAdaptionTime'])
POST_ADAPTION_TIME = float(parser['ExperimentOptions']['PostAdaptionTime'])
TEST_STIMULUS_TIME = float(parser['ExperimentOptions']['TestStimulusTime'])
ISI_TIME = float(parser['ExperimentOptions']['ISITime'])

# GRAPHICAL OPTIONS
BG_COLOR = '#5E5E5E'

# right = CW, left = CCW
key_list = ['right', 'left']

data_columns = ['Response', 'Test Stim', 'Test Speed', 'Correct', 'Time']
experiment_info_columns = ['Subject', 'Age', 'Gender', 'Experimenter', 'Adapting', 'View Dist (cm)', 'Width (cm)', 'Width (px)', 'Height (px)', 'Refresh Rate (Hz)', 'Timestamp', 'Adaption Dir']

# Global variables
window = None
fixator = None
prompt = None
subject = None
run_practice = None
timestamp = None
adaption_dir = None

stimTypes = None
stimLog, stimLog_t, stimMirror, stimMirror_t = None, None, None, None

def main():
    global parser, window, fixator, prompt, subject, run_practice, timestamp, adaption_dir
    global stimTypes, stimLog, stimLog_t, stimMirror, stimMirror_t

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    key_list.append('escape')

    # Prompt GUI for experimental setup
    dlg = gui.Dlg(title='Experiment Setup')
    dlg.addField('Participant ID:')                                                         # exp_res[0]
    dlg.addField('Participant Age:')                                                        # exp_res[1]
    dlg.addField('Participant Gender:', choices=['M', 'F'])                                 # exp_res[2]
    dlg.addField('Experimenter ID:')                                                        # exp_res[3]
    dlg.addField('Adaption:', 'Both', choices=['Both', 'No Adaptation Only', 'Adaptation Only'])     # exp_res[4]
    dlg.addField('Number of Unique Stimuli:', '2', choices=['2', '3'])                      # exp_res[5]
    dlg.addField('Practice for No Adaption:', 'Yes', choices=['Yes', 'No'])                 # exp_res[6]
    dlg.addField('Adaption Direction:', 'Both', choices=['Both', 'CW', 'CCW'])              # exp_res[7]
    exp_res = dlg.show()
    if not dlg.OK:
        exit()

    def_width_cm, def_width_px, def_height_px = '', '', ''
    if os.path.exists(f'data/experiment_info.csv'):
        exp_df = pd.read_csv('data/experiment_info.csv', index_col=0)

        # Used for compatability with previous version that didn't have specified adaption direction implemented
        if 'Adaption Dir' not in exp_df:
            exp_df['Adaption Dir'] = ''

        rows = exp_df.loc[exp_df['Subject'] == exp_res[0]]
        if len(rows) > 0:
            def_width_cm = str(rows.iloc[0]['Width (cm)'])
            def_width_px = str(rows.iloc[0]['Width (px)'])
            def_height_px = str(rows.iloc[0]['Height (px)'])
            dlg = gui.Dlg(title='Subject Settings Found')
            dlg.addText('Subject ID with monitor info has been found.')
            dlg.show()

    # Parse experimentation info about practice runs, stimulus types, adaption direction
    run_practice = True if exp_res[6] == 'Yes' else False
    numStim = int(exp_res[5])
    stimTypes = ['log', 'mirror']
    if numStim == 3:
        stimTypes.append('both')
    adaption_dir = exp_res[7]
    if exp_res[4] == 'No Adaptation Only':
        adaption_dir = ''

    exp_res = exp_res[:5]

    # Prompt GUI for monitor setup
    dlg = gui.Dlg(title='Monitor Setup')
    dlg.addText('Monitor Info:')
    dlg.addField('Viewing distance (cm):', '57')
    dlg.addField('Width (cm):', def_width_cm)
    dlg.addField('Width (pixels):', def_width_px)
    dlg.addField('Height (pixels):', def_height_px)

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
    to_run_no_adaption = True if exp_res[4] != 'Adaptation Only' else False
    to_run_adaption = True if exp_res[4] != 'No Adaptation Only' else False

    # Create monitor profile and window for experiment
    monitor = monitors.Monitor('monitor')
    monitor.setSizePix((height_px, width_px))
    monitor.setWidth(width_cm)
    monitor.setDistance(viewing_dist)
    window = visual.Window(color=BG_COLOR, monitor=monitor, fullscr=True) 
    window.mouseVisible = False

    '''
    gamma_table = np.load('gammaTable.npy')
    gamma_ramp = np.vstack((gamma_table, gamma_table, gamma_table))
    window.gammaRamp = gamma_ramp
    '''

    # Parse refresh rate of monitor
    refresh_rate = window.getActualFrameRate()
    if refresh_rate is None:
        refresh_rate = 'N/A'
    else:
        refresh_rate = str(round(refresh_rate))
    fixator = visual.Circle(window, size=(FIXATOR_SIZE_PX, FIXATOR_SIZE_PX), units='pix')

    # Save experiment setup data to file
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/experiment_info.csv'):
        df = pd.DataFrame(columns=experiment_info_columns)
    else:
        df = pd.read_csv('data/experiment_info.csv', index_col=0)
    
    df = df.append(pd.Series(exp_res + monitor_info + [refresh_rate, timestamp, adaption_dir], index=experiment_info_columns), ignore_index=True)
    df.to_csv('data/experiment_info.csv')

    # Generate stimuli
    stimLog_t = generate_stimulus()
    stimMirror_t = generate_stimulus(dir=-1)
    stimLog = generate_stimulus(transparent=False)
    stimMirror = generate_stimulus(transparent=False, dir=-1)

    # Run experiment parts
    if to_run_no_adaption:
        run_without_adaption(window)
    if to_run_adaption:
        if to_run_no_adaption and to_run_adaption:
            prompt.text = 'You have finished the first part of the experiment. We will now move on to the second part.\n\nPress any key to continue.'
            prompt.draw()
            window.flip()
            event.waitKeys()
        run_with_adaption(window)

    quit(True)

def run_without_adaption(window):
    global parser, prompt, subject
    data = pd.DataFrame(columns=data_columns)

    # Experimental instructions
    prompt = visual.TextStim(window, color='white', text=f'You will see a blank screen for {round(INIT_NO_ADAPTION_TIME)} seconds. You do not have to press anything. Try to keep your eyes fixed during this time.\n\nThen, you will hear a beep to alert you that a test is about to begin.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()
    prompt.text = f'The test will be a rotating spiral. Press the right arrow key if it appears to be rotating to the right (clockwise) and the left arrow key if it appears to be rotating to the left (counterclockwise).\n\nAfter the spiral, you\'\'ll see a blank screen again for {round(POST_NO_ADAPTION_TIME)} seconds, and then a beep and test image.\n\nThis will repeat until the end of the experiment.\n\nPress any key to continue.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    # Generate stimulus speeds
    min_speed = float(parser['NoAdaptionOptions']['MinTestSpeed'])
    max_speed = float(parser['NoAdaptionOptions']['MaxTestSpeed'])
    step_size = float(parser['NoAdaptionOptions']['StepSize'])
    speeds = [round(n, 5) for n in np.arange(min_speed, max_speed + step_size, step=step_size)]

    # Run through practice trials if specified
    if run_practice:
        prompt.text = f'You will start with a practice round where the tests will get progressively more difficult. You will hear a beep if your response is incorrect. \n\nPress any key to start.'
        prompt.draw()
        window.flip()
        event.waitKeys()

        # Initialize parameters for practice trials
        num_blocks = int(parser['PracticeOptions']['NumBlocks'])
        min_per_block = int(parser['PracticeOptions']['MinTrialsPerBlock'])
        acc_thresh = float(parser['PracticeOptions']['AccuracyThreshold'])
        starting_trial_time = float(parser['PracticeOptions']['StartingTrialTime'])
        practice_isi_time = float(parser['PracticeOptions']['ISITime'])

        show_times = np.linspace(starting_trial_time, TEST_STIMULUS_TIME, num=num_blocks)
        practice_data = pd.DataFrame(columns=data_columns+['Stim Time'])

        # Iterate through different stimulus times
        for stim_time in show_times:
            num_correct, trial_count = 0, 0

            # Iterate through same stimulus time until accuracy threshold met
            while True:
                if trial_count > min_per_block and num_correct/trial_count >= acc_thresh:
                    break
                trial_speed = random.choice(speeds)
                trial_stim = random.choice(stimTypes)
                
                fixator.color = 'red'
                fixator.draw()
                window.flip()

                # Top-up adaptor (blank screen)
                start_time = time.time()
                while time.time() - start_time < practice_isi_time:
                    if len(event.getKeys(keyList=['escape'])) > 0:
                        quit()

                fixator.color = 'green'
                event.clearEvents()

                res = []
                reset_stims()

                # Display practice test stimulus
                start_time = time.time()
                while len(res) == 0 and time.time() - start_time < stim_time:
                    res = event.getKeys(keyList=key_list)
                    if 'escape' in res:
                        quit()
                    
                    if trial_stim == 'log':
                        stimLog.draw()
                        stimLog.ori = stimLog.ori + trial_speed/100
                    elif trial_stim == 'mirror':
                        stimMirror.draw()
                        stimMirror.ori = stimMirror.ori + trial_speed/100
                    else:
                        stimLog.draw()
                        stimMirror_t.draw()
                        stimLog.ori = stimLog.ori + trial_speed/100
                        stimMirror_t.ori = stimMirror_t.ori + trial_speed/100

                    fixator.draw()
                    window.flip()

                    core.wait(0.01)

                # Clear window and wait for response if one hasn't been given yet
                window.flip()
                if len(res) == 0:
                    res = event.waitKeys()
                if 'escape' in res:
                    quit()

                # Update current block accuracy, provide audio feedback if incorrect
                end_time = time.time()
                if (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right'):
                    num_correct += 1
                trial_count += 1

                is_correct = (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right')
                if not is_correct:
                    beep = sound.Sound('A', secs=0.2)
                    beep.play()

                # Update and save practice data
                practice_data = practice_data.append(pd.Series([res[0], trial_stim, trial_speed, is_correct, end_time - start_time, stim_time], index=practice_data.columns), ignore_index=True)
                practice_data.to_csv(f'data/{subject}{timestamp}_practice.csv')

        prompt.text = f'We will now move on to the actual experiment.\n\nPress any key to start.'
        prompt.draw()
        window.flip()
        event.waitKeys()

    # Generate trials based on combinations of stimulus type and speed
    trials = []
    combinations = []
    for stim in stimTypes:
        for speed in speeds:
            combinations.append({'stim': stim, 'speed': speed})

    [trials.extend(combinations) for _ in range(TRIALS_PER_COMBINATION)]
    random.shuffle(trials)

    # Iterate through trials one by one
    is_first_trial = True
    for trial in trials:
        trial_speed = trial['speed']
        trial_stim = trial['stim']

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
        if trial_stim == 'log':
            stimLog.draw()
        elif trial_stim == 'mirror':
            stimMirror.draw()
        else:
            stimLog.draw()
            stimMirror_t.draw()
        fixator.draw()
        window.flip()
        core.wait(ISI_TIME)
        beep = sound.Sound('A', secs=0.2)
        beep.play()
        fixator.color = 'green'
        if trial_stim == 'log':
            stimLog.draw()
        elif trial_stim == 'mirror':
            stimMirror.draw()
        else:
            stimLog.draw()
            stimMirror_t.draw()
        fixator.draw()
        window.flip()

        # Display est stimulus
        res = []
        event.clearEvents()
        reset_stims()
        start_time = time.time()
        while len(res) == 0 and time.time() - start_time < TEST_STIMULUS_TIME:
            res = event.getKeys(keyList=key_list)

            if 'escape' in res:
                quit()

            if trial_stim == 'log':
                stimLog.draw()
                stimLog.ori = stimLog.ori + trial_speed/100
            elif trial_stim == 'mirror':
                stimMirror.draw()
                stimMirror.ori = stimMirror.ori + trial_speed/100
            else:
                stimLog.draw()
                stimMirror_t.draw()
                stimLog.ori = stimLog.ori + trial_speed/100
                stimMirror_t.ori = stimMirror_t.ori + trial_speed/100

            fixator.draw()
            window.flip()
            core.wait(0.01)

        # Clear window and wait for response if one hasn't been given yet
        window.flip()
        if len(res) == 0:
            res = event.waitKeys()
        if 'escape' in res:
            quit()

        # Update and save data with psychometric plot
        end_time = time.time()
        is_correct = (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right')
        data = data.append(pd.Series([res[0], trial_stim, trial_speed, is_correct, end_time - start_time], index=data.columns), ignore_index=True)
        data.to_csv(f'data/{subject}{timestamp}_noAdapt.csv')
        save_psychometric_plot(data, 'No', 'noAdapt')

def run_with_adaption(window):
    global parser, prompt, subject
    data = pd.DataFrame(columns=data_columns)

    # Experimental instructions
    prompt = visual.TextStim(window, color='white', text=f'You will first see a rotating spiral for {round(INIT_ADAPTION_TIME)} seconds. Try to keep your eyes fixed on the red dot during this time.\n\nPress any key to continue.')
    prompt.draw()
    window.flip()
    event.waitKeys()
    prompt.text = f'The test will be a rotating spiral. Press the right arrow key if it appears to be rotating to the right (clockwise) and the left arrow key if it appears to be rotating to the left (counterclockwise).\n\nAfter the spiral, you\'\'ll see a rotating spiral again for {round(POST_ADAPTION_TIME)} seconds, and then a beep and test spiral.\n\nThis will repeat until the end of the experiment.\n\nPress any key to start.'
    prompt.draw()
    window.flip()
    event.waitKeys()

    # Generate adaptor speeds
    min_speed = float(parser['AdaptionOptions']['MinTestSpeed'])
    max_speed = float(parser['AdaptionOptions']['MaxTestSpeed'])
    step_size = float(parser['AdaptionOptions']['StepSize'])
    speeds = [round(n, 5) for n in np.arange(min_speed, max_speed + step_size, step=step_size)]

    # Generate trials based on combinations of stimulus type and speed
    trials = []
    combinations = []
    for stim in stimTypes:
        for speed in speeds:
            combinations.append({'stim': stim, 'speed': speed})
    [trials.extend(combinations) for _ in range(TRIALS_PER_COMBINATION)]
    random.shuffle(trials)

    # Set up adaption speeds based on adaption direction specified
    if adaption_dir == 'Both':
        log_adaptor_speed = float(parser['AdaptionOptions']['DualAdaptorSpeed'])
        mirror_adaptor_speed = -log_adaptor_speed
        adaptor_dir_change_prop = float(parser['AdaptionOptions']['AdaptorDirChangeTime'])
    elif adaption_dir == 'CW':
        log_adaptor_speed = float(parser['AdaptionOptions']['CWAdaptorSpeed'])
        mirror_adaptor_speed = log_adaptor_speed
        adaptor_dir_change_prop = 1.0
    else:
        log_adaptor_speed = float(parser['AdaptionOptions']['CCWAdaptorSpeed'])
        mirror_adaptor_speed = log_adaptor_speed
        adaptor_dir_change_prop = 1.0

    print(log_adaptor_speed)
    print(mirror_adaptor_speed)
    # Iterate through trials one by one
    is_first_trial = True
    for trials in trials:
        trial_stim = trials['stim']
        trial_speed = trials['speed']

        fixator.color = 'red'
        adaption_time = INIT_ADAPTION_TIME if is_first_trial else POST_ADAPTION_TIME
        if is_first_trial:
            is_first_trial = False

        # Top-up adaptor
        start_time = time.time()
        while time.time() - start_time < adaption_time:
            if len(event.getKeys(keyList=['escape'])) > 0:
                quit()

            stimLog.draw()
            stimMirror_t.draw()

            if time.time() - start_time < adaptor_dir_change_prop * adaption_time:
                stimLog.ori = stimLog.ori + log_adaptor_speed/100
                stimMirror_t.ori = stimMirror_t.ori + mirror_adaptor_speed/100
            else:
                stimLog.ori = stimLog.ori - log_adaptor_speed/100
                stimMirror_t.ori = stimMirror_t.ori - mirror_adaptor_speed/100

            if int(stimLog.ori) % 360 == 0:
                print(time.time() - start_time)

            fixator.draw()
            window.flip()
            core.wait(0.01)

        # ISI
        if trial_stim == 'log':
            stimLog.draw()
        elif trial_stim == 'mirror':
            stimMirror.draw()
        else:
            stimLog.draw()
            stimMirror_t.draw()
        fixator.draw()
        window.flip()
        core.wait(ISI_TIME)
        beep = sound.Sound('A', secs=0.2)
        beep.play()
        fixator.color = 'green'
        if trial_stim == 'log':
            stimLog.draw()
        elif trial_stim == 'mirror':
            stimMirror.draw()
        else:
            stimLog.draw()
            stimMirror_t.draw()
        fixator.draw()
        window.flip()

        # Display test stimulus
        res = []
        event.clearEvents()
        start_time = time.time()
        while len(res) == 0 and time.time() - start_time < TEST_STIMULUS_TIME:
            res = event.getKeys(keyList=key_list)
            if 'escape' in res:
                quit()

            if trial_stim == 'log':
                stimLog.draw()
                stimLog.ori = stimLog.ori + trial_speed/100
            elif trial_stim == 'mirror':
                stimMirror.draw()
                stimMirror.ori = stimMirror.ori + trial_speed/100
            else:
                stimLog.draw()
                stimMirror_t.draw()
                stimLog.ori = stimLog.ori + trial_speed/100
                stimMirror_t.ori = stimMirror_t.ori + trial_speed/100

            fixator.draw()
            window.flip()
            core.wait(0.01)

        # Clear window and wait for response if one hasn't been given yet
        window.flip()
        if len(res) == 0:
            res = event.waitKeys()
        if 'escape' in res:
            quit()

        # Save and update data with psychometric plot
        end_time = time.time()
        is_correct = (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right')
        data = data.append(pd.Series([res[0], trial_stim, trial_speed, is_correct, end_time - start_time], index=data.columns), ignore_index=True)
        data.to_csv(f'data/{subject}{timestamp}_Adapt.csv')
        save_psychometric_plot(data, 'With', 'Adapt')

def generate_stimulus(transparent = True, dir = 1):
    
    # Coefficient that determines how spaced the lines are
    b = 0.7

    radius = np.exp(2*np.pi*b)
    lw = float(parser['StimulusOptions']['StimLineWidth'])
    
    phase_number = 1
    phase_thresh = np.pi / NUM_PHASES
    t = np.linspace(0, 2*np.pi, 100)
    phi_vals = np.linspace(0, 2*np.pi, 1000)

    # Build outline of image
    if transparent:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(facecolor=BG_COLOR)
        circle = plt.Circle((0, 0), radius, edgecolor=None, facecolor=BG_COLOR, fill=True)        

    # Draw black or white spirals depending on phi value around circle
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

        ax.plot(x_n, y_n, color=c, linewidth=lw, linestyle='--')

    # Set graphical options to remove axes, set aspect ratio
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.axis('off')

    # Save figure as image to use as stimulus
    if transparent:
        plt.savefig('stim.png', bbox_inches='tight', transparent = True)
    else:
        ax.add_artist(circle)
        plt.savefig('stim.png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    
    stim = visual.ImageStim(window, image='stim.png', units='deg', size=(STIM_SIZE, STIM_SIZE))
    os.remove('stim.png')
    return stim

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

def reset_stims():
    stimLog_t.ori = 0
    stimMirror_t.ori = 0
    stimLog.ori = 0
    stimMirror.ori = 0

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
    plt.title(f'Subject {subject}{timestamp} ({adapt} Adapt)')
    plt.xlabel('Test Stimulus Speed (deg/s)')
    plt.ylabel('Proportion of Responding Clockwise')
    plt.savefig(f'data/{subject}{timestamp}_{adaptName}_plot.png')

if __name__ == "__main__":
    main()
