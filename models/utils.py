import pandas as pd
from scipy import signal


def check_for_time_values_error(Y, T, interval):
    # If truthy, T must be the same length as Y, and interval must also be truthy
    if T is not None:
        if len(Y) != len(T):
            raise Exception('Provided times should have same length as labels')
        if not interval:
            raise Exception('A window length must be provided when using label times to train hmm')


def restore_labels_after_gaps(y_pred, y_smooth, t, interval):
    # Restore unsmoothed predictions to labels following gaps in time
    df = pd.DataFrame({'y_pred': y_pred, 'y_smooth': y_smooth})

    if type(t[0]) == int:
        gaps = (pd.Series(t).diff(periods=1) != interval)
        gaps[0] = False
    else:
        gaps = (pd.Series(t).diff(periods=1) != pd.Timedelta(seconds=interval))
        gaps[0] = False

    df.loc[gaps, 'y_smooth'] = df.loc[gaps, 'y_pred']

    return df['y_smooth'].values


def calculate_transition_matrix(Y, t=None, interval=None):
    # t and interval are used to identify any gaps in the data
    # If not provided, it is assumed there are no gaps
    check_for_time_values_error(Y, t, interval)
    
    t = t if t is not None else range(len(Y))
    interval = interval or 1

    df = pd.DataFrame(Y)

    # create a new column with data shifted one space
    df['shift'] = df[0].shift(-1)

    # only consider transitions of expected interval
    if type(t[0]) == int:
        df = df[(-pd.Series(t).diff(periods=-1) == interval)]
    else:
        df = df[(-pd.Series(t).diff(periods=-1) == pd.Timedelta(seconds=interval))]

    # add a count column (for group by function)
    df['count'] = 1
    
    # groupby and then unstack, fill the zeros
    trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
    
    # normalise by occurences and save values to get the transition matrix
    return trans_mat.div(trans_mat.sum(axis=1), axis=0).values


def butterfilt(x, cutoffs, fs, order=10, axis=0):
    """ Butterworth filter """
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        hicut, lowcut = cutoffs
        if hicut > 0:
            btype = 'bandpass'
            Wn = (hicut / nyq, lowcut / nyq)
        else:
            btype = 'low'
            Wn = lowcut / nyq
    else:
        btype = 'low'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    return y
