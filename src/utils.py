import os
import sys

import numpy as np
import time

import matplotlib.pyplot as plt

from torchvision.utils import make_grid


##### data loader utils #####
def unpickle(file):
    '''
    take in python pickled input file and return dictionary
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_training_helper(dir_path, file_name):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        train_dict = unpickle(os.path.join(dir_path, '{0}_{1}'.format(file_name, i)))
        if i == 1:
            train_data = train_dict[b'data']
        else:
            train_data = np.vstack((train_data, train_dict[b'data']))

        train_labels += train_dict[b'labels']

    train_data = np.rollaxis(train_data.reshape(train_data.shape[0], 3, 32, 32), 1, 4)
    train_labels = np.array(train_labels)
    return train_data, train_labels

def load_testing_helper(dir_path, file_name):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        train_dict = unpickle(os.path.join(dir_path, 'test_batch'))
        if i == 1:
            train_data = train_dict[b'data']
        else:
            train_data = np.vstack((train_data, train_dict[b'data']))

        train_labels += train_dict[b'labels']

    train_data = np.rollaxis(train_data.reshape(train_data.shape[0], 3, 32, 32), 1, 4)
    train_labels = np.array(train_labels)
    return train_data, train_labels
    
def load_data(data_dir, return_label_name = False):
    
    train_data, train_labels = load_training_helper(data_dir, 'data_batch')
    
    test_data, test_labels = load_testing_helper(data_dir, 'test_batch')
    if return_label_name:
        meta_data_dict = unpickle(data_dir + "/batches.meta")
        label_names = meta_data_dict[b'label_names']
        return train_data, train_labels, test_data, test_labels, return_label_name
    else:
        return train_data, train_labels, test_data, test_labels
    
    
##### display images #####  
##### reference: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py#L94 #####
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = os.popen('stty size', 'r').read().split()
term_width = 10      
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
    
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



