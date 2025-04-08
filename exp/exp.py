import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psychopy import core, event, gui, monitors, sound, visual
from utils.constants import (
    ADAPTION_ADAPTOR_DIR_CHANGE_TIME,
    ADAPTION_CCW_ADAPTOR_SPEED,
    ADAPTION_CW_ADAPTOR_SPEED,
    ADAPTION_DUAL_ADAPTOR_SPEED,
    ADAPTION_MAX_TEST_SPEED,
    ADAPTION_MIN_TEST_SPEED,
    ADAPTION_STEP_SIZE,
    BG_COLOR,
    FIXATOR_SIZE_PX,
    INIT_ADAPTION_TIME,
    INIT_NO_ADAPTION_TIME,
    ISI_TIME,
    KEY_ADAPTION_DIRECTION,
    KEY_ADAPTION_TYPE,
    KEY_EXPERIMENTER_ID,
    KEY_PARTICIPANT_AGE,
    KEY_PARTICIPANT_GENDER,
    KEY_PARTICIPANT_ID,
    KEY_PRACTICE_NO_ADAPTION,
    KEY_UNIQUE_STIMULI,
    NO_ADAPTION_MAX_TEST_SPEED,
    NO_ADAPTION_MIN_TEST_SPEED,
    NO_ADAPTION_STEP_SIZE,
    NUM_PHASES,
    POST_ADAPTION_TIME,
    POST_NO_ADAPTION_TIME,
    PRACTICE_ACCURACY_THRESHOLD,
    PRACTICE_ISI_TIME,
    PRACTICE_MIN_TRIALS_PER_BLOCK,
    PRACTICE_NUM_BLOCKS,
    PRACTICE_STARTING_TRIAL_TIME,
    STIM_LINE_WIDTH,
    STIM_SIZE,
    TEST_STIMULUS_TIME,
    TRIALS_PER_COMBINATION,
)
from utils.stimulus import generate_stimulus

random.seed(time.time())


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
    global window, fixator, prompt, subject, run_practice, timestamp, adaption_dir
    global stimTypes, stimLog, stimLog_t, stimMirror, stimMirror_t

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    key_list.append('escape')

    # Prompt GUI for experimental setup
    dlg = gui.Dlg(title='Experiment Setup')
    dlg.addField(KEY_PARTICIPANT_ID)
    dlg.addField(KEY_PARTICIPANT_AGE)
    dlg.addField(KEY_PARTICIPANT_GENDER, choices=['M', 'F'])
    dlg.addField(KEY_EXPERIMENTER_ID)
    dlg.addField(KEY_ADAPTION_TYPE, 'Both', choices=['Both', 'No Adaptation Only', 'Adaptation Only'])
    dlg.addField(KEY_UNIQUE_STIMULI, '2', choices=['2', '3'])
    dlg.addField(KEY_PRACTICE_NO_ADAPTION, 'Yes', choices=['Yes', 'No'])
    dlg.addField(KEY_ADAPTION_DIRECTION, 'Both', choices=['Both', 'CW', 'CCW'])
    exp_res = dlg.show()
    if not dlg.OK:
        exit()

    def_width_cm, def_width_px, def_height_px = '', '', ''
    if os.path.exists('data/experiment_info.csv'):
        exp_df = pd.read_csv('data/experiment_info.csv', index_col=0)

        # Used for compatability with previous version that didn't have specified adaption direction implemented
        if 'Adaption Dir' not in exp_df:
            exp_df['Adaption Dir'] = ''

        rows = exp_df.loc[exp_df['Subject'] == exp_res[KEY_PARTICIPANT_ID]]
        if len(rows) > 0:
            def_width_cm = str(rows.iloc[0]['Width (cm)'])
            def_width_px = str(rows.iloc[0]['Width (px)'])
            def_height_px = str(rows.iloc[0]['Height (px)'])
            dlg = gui.Dlg(title='Subject Settings Found')
            dlg.addText('Subject ID with monitor info has been found.')
            dlg.show()

    # Parse experimentation info about practice runs, stimulus types, adaption direction
    run_practice = True if exp_res[KEY_PRACTICE_NO_ADAPTION] == 'Yes' else False
    numStim = int(exp_res[KEY_UNIQUE_STIMULI])
    stimTypes = ['log', 'mirror']
    if numStim == 3:
        stimTypes.append('both')
    adaption_dir = exp_res[KEY_ADAPTION_DIRECTION]
    if exp_res[KEY_ADAPTION_TYPE] == 'No Adaptation Only':
        adaption_dir = ''

    # Prompt GUI for monitor setup
    monitor_info = []
    while True:
        dlg = gui.Dlg(title='Monitor Setup')
        dlg.addText('Please fill out all fields and enter integer values only.')
        dlg.addText('Monitor Info:')
        dlg.addField('Viewing distance (cm):', '57')
        dlg.addField('Width (cm):', def_width_cm)
        dlg.addField('Width (pixels):', def_width_px)
        dlg.addField('Height (pixels):', def_height_px)

        monitor_info = dlg.show().values()
        if not dlg.OK:
            exit()
        if '' not in monitor_info and all([s.isdigit() for s in monitor_info]):
            break

    viewing_dist, width_cm, width_px, height_px = [int(n) for n in monitor_info]

    subject = exp_res[KEY_PARTICIPANT_ID]
    to_run_no_adaption = True if exp_res[KEY_ADAPTION_TYPE] != 'Adaptation Only' else False
    to_run_adaption = True if exp_res[KEY_ADAPTION_TYPE] != 'No Adaptation Only' else False

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

    entry = pd.Series(list(exp_res.values()) + list(monitor_info) + [refresh_rate, timestamp, adaption_dir])
    df.loc[len(df)] = entry
    df.to_csv('data/experiment_info.csv')

    # Generate stimuli
    common_opts = {
        'size': STIM_SIZE,
        'linewidth': STIM_LINE_WIDTH,
        'n_phase': NUM_PHASES,
        'bg_color': BG_COLOR
    }

    stimLog_t = generate_stimulus(window, **common_opts)
    stimMirror_t = generate_stimulus(window, dir=-1, **common_opts)
    stimLog = generate_stimulus(window, transparent=False, **common_opts)
    stimMirror = generate_stimulus(window, transparent=False, dir=-1, **common_opts)

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
    global prompt, subject
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
    min_speed = NO_ADAPTION_MIN_TEST_SPEED
    max_speed = NO_ADAPTION_MAX_TEST_SPEED
    step_size = NO_ADAPTION_STEP_SIZE
    speeds = [round(n, 5) for n in np.arange(min_speed, max_speed + step_size, step=step_size)]

    # Run through practice trials if specified
    if run_practice:
        prompt.text = 'You will start with a practice round where the tests will get progressively more difficult. You will hear a beep if your response is incorrect. \n\nPress any key to start.'
        prompt.draw()
        window.flip()
        event.waitKeys()

        # Initialize parameters for practice trials
        num_blocks = PRACTICE_NUM_BLOCKS
        min_per_block = PRACTICE_MIN_TRIALS_PER_BLOCK
        acc_thresh = PRACTICE_ACCURACY_THRESHOLD
        starting_trial_time = PRACTICE_STARTING_TRIAL_TIME
        practice_isi_time = PRACTICE_ISI_TIME

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
                clock = core.MonotonicClock()
                while clock.getTime() < practice_isi_time:
                    if len(event.getKeys(keyList=['escape'])) > 0:
                        quit()

                fixator.color = 'green'
                event.clearEvents()

                res = []
                reset_stims()

                # Display practice test stimulus
                clock = core.MonotonicClock()
                while len(res) == 0 and clock.getTime() < stim_time:
                    res = event.getKeys(keyList=key_list)
                    if 'escape' in res:
                        quit()

                    t = clock.getTime()
                    if trial_stim == 'log':
                        stimLog.draw()
                        stimLog.ori = t * trial_speed
                    elif trial_stim == 'mirror':
                        stimMirror.draw()
                        stimMirror.ori = t * trial_speed
                    else:
                        stimLog.draw()
                        stimMirror_t.draw()
                        stimLog.ori = t * trial_speed
                        stimMirror_t.ori = t * stimMirror_t.ori

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
                end_time = clock.getTime()
                if (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right'):
                    num_correct += 1
                trial_count += 1

                is_correct = (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right')
                if not is_correct:
                    beep = sound.Sound('A', secs=0.2, stereo=True)
                    beep.play()

                # Update and save practice data
                entry = [res[0], trial_stim, trial_speed, is_correct, end_time, stim_time]
                practice_data.loc[len(practice_data)] = entry
                practice_data.to_csv(f'data/{subject}{timestamp}_practice.csv')

        prompt.text = 'We will now move on to the actual experiment.\n\nPress any key to start.'
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
        clock = core.MonotonicClock()
        while clock.getTime() < wait_time:
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
        beep = sound.Sound('A', secs=0.2, stereo=True)
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
        clock = core.MonotonicClock()
        while len(res) == 0 and clock.getTime() < TEST_STIMULUS_TIME:
            res = event.getKeys(keyList=key_list)

            if 'escape' in res:
                quit()

            t = clock.getTime()
            if trial_stim == 'log':
                stimLog.draw()
                stimLog.ori = t * trial_speed
            elif trial_stim == 'mirror':
                stimMirror.draw()
                stimMirror.ori = t * trial_speed
            else:
                stimLog.draw()
                stimMirror_t.draw()
                stimLog.ori = t * trial_speed
                stimMirror_t.ori = t * trial_speed

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
        end_time = clock.getTime()
        is_correct = (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right')
        entry = [res[0], trial_stim, trial_speed, is_correct, end_time]
        data.loc[len(data)] = entry
        data.to_csv(f'data/{subject}{timestamp}_noAdapt.csv')
        save_psychometric_plot(data, 'No', 'noAdapt')

def run_with_adaption(window):
    global prompt, subject
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
    min_speed = ADAPTION_MIN_TEST_SPEED
    max_speed = ADAPTION_MAX_TEST_SPEED
    step_size = ADAPTION_STEP_SIZE
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
        log_adaptor_speed = ADAPTION_DUAL_ADAPTOR_SPEED
        mirror_adaptor_speed = -log_adaptor_speed
        adaptor_dir_change_prop = ADAPTION_ADAPTOR_DIR_CHANGE_TIME
    elif adaption_dir == 'CW':
        log_adaptor_speed = ADAPTION_CW_ADAPTOR_SPEED
        mirror_adaptor_speed = log_adaptor_speed
        adaptor_dir_change_prop = 1.0
    else:
        log_adaptor_speed = ADAPTION_CCW_ADAPTOR_SPEED
        mirror_adaptor_speed = log_adaptor_speed
        adaptor_dir_change_prop = 1.0

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
        clock = core.MonotonicClock()
        while clock.getTime() < adaption_time:
            if len(event.getKeys(keyList=['escape'])) > 0:
                quit()

            stimLog.draw()
            stimMirror_t.draw()

            t = clock.getTime()
            if t < adaptor_dir_change_prop * adaption_time:
                stimLog.ori = t * log_adaptor_speed
                stimMirror_t.ori = t * mirror_adaptor_speed
            else:
                stimLog.ori = t * (-log_adaptor_speed)
                stimMirror_t.ori = t * (-mirror_adaptor_speed)

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
        beep = sound.Sound('A', secs=0.2, stereo=True)
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
        clock = core.MonotonicClock()
        while len(res) == 0 and clock.getTime() < TEST_STIMULUS_TIME:
            res = event.getKeys(keyList=key_list)
            if 'escape' in res:
                quit()

            t = clock.getTime()
            if trial_stim == 'log':
                stimLog.draw()
                stimLog.ori = t * trial_speed
            elif trial_stim == 'mirror':
                stimMirror.draw()
                stimMirror.ori = t * trial_speed
            else:
                stimLog.draw()
                stimMirror_t.draw()
                stimLog.ori = t * trial_speed
                stimMirror_t.ori = t * trial_speed

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
        end_time = clock.getTime()
        is_correct = (trial_speed < 0 and res[0] == 'left') or (trial_speed > 0 and res[0] == 'right')

        entry = [res[0], trial_stim, trial_speed, is_correct, end_time]
        data.loc[len(data)] = entry
        data.to_csv(f'data/{subject}{timestamp}_Adapt.csv')
        save_psychometric_plot(data, 'With', 'Adapt')

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
