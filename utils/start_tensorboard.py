
" START TENSORBOARD "
import os, glob
from tensorboard import program

import time
program.logger.setLevel('INFO')

def run_tensorboard(new_run):

    path = os.getcwd() + '/lightning_logs/'

    try:
        newest_folder = max(glob.glob(os.path.join(path, '*/')), key=os.path.getmtime)
        version_number = newest_folder.split('\\')[-2].split('_')[1]

        if new_run:
            new_version_number = str(int(version_number) + 1)
            newest_folder = newest_folder.replace(version_number, new_version_number)

            version_number = new_version_number # for print purposes

    except ValueError:
        version_number = 0
        newest_folder = path + 'version_0'

    while not os.path.exists(newest_folder):
        time.sleep(1)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', newest_folder])
    url = tb.launch()

    print('\n')
    print('-' * 20)
    print('Starting tensorboard at URL %s version %s' % (url, version_number))
    print('-' * 20)
    print('\n')

    while True: # to keep script alive
        time.sleep(2)

if __name__ == '__main__':
    run_tensorboard(new_run=False)