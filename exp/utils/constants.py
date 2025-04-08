import configparser

parser = configparser.ConfigParser()
parser.read('config.ini')

# EXPERIMENTAL SETTINGS
STIM_LINE_WIDTH = float(parser['StimulusOptions']['StimLineWidth'])
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

NO_ADAPTION_MIN_TEST_SPEED = float(parser['NoAdaptionOptions']['MinTestSpeed'])
NO_ADAPTION_MAX_TEST_SPEED = float(parser['NoAdaptionOptions']['MaxTestSpeed'])
NO_ADAPTION_STEP_SIZE = float(parser['NoAdaptionOptions']['StepSize'])

PRACTICE_NUM_BLOCKS = int(parser['PracticeOptions']['NumBlocks'])
PRACTICE_MIN_TRIALS_PER_BLOCK = int(parser['PracticeOptions']['MinTrialsPerBlock'])
PRACTICE_ACCURACY_THRESHOLD = float(parser['PracticeOptions']['AccuracyThreshold'])
PRACTICE_STARTING_TRIAL_TIME = float(parser['PracticeOptions']['StartingTrialTime'])
PRACTICE_ISI_TIME = float(parser['PracticeOptions']['ISITime'])

ADAPTION_MIN_TEST_SPEED = float(parser['AdaptionOptions']['MinTestSpeed'])
ADAPTION_MAX_TEST_SPEED = float(parser['AdaptionOptions']['MaxTestSpeed'])
ADAPTION_STEP_SIZE = float(parser['AdaptionOptions']['StepSize'])
ADAPTION_DUAL_ADAPTOR_SPEED = float(parser['AdaptionOptions']['DualAdaptorSpeed'])
ADAPTION_ADAPTOR_DIR_CHANGE_TIME = float(parser['AdaptionOptions']['AdaptorDirChangeTime'])
ADAPTION_CW_ADAPTOR_SPEED = float(parser['AdaptionOptions']['CWAdaptorSpeed'])
ADAPTION_CCW_ADAPTOR_SPEED = float(parser['AdaptionOptions']['CCWAdaptorSpeed'])

# GRAPHICAL OPTIONS
BG_COLOR = '#5E5E5E'

# Other constants
KEY_PARTICIPANT_ID = 'Participant ID'
KEY_PARTICIPANT_AGE = 'Participant Age'
KEY_PARTICIPANT_GENDER = 'Participant Gender'
KEY_EXPERIMENTER_ID = 'Experimenter ID'
KEY_ADAPTION_TYPE = 'Adaption'
KEY_UNIQUE_STIMULI = 'Number of Unique Stimuli'
KEY_PRACTICE_NO_ADAPTION = 'Practice for No Adaption'
KEY_ADAPTION_DIRECTION = 'Adaption Direction'
