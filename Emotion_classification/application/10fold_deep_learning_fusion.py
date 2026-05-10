import random
import sys, os, argparse
import numpy as np
import torch


sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ubuntu kill all python
# ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

def main(argv):
    model_name = ['CNN_LSTM', 'CNN_Transformer', 'CNN_GRU', 'MTRCNN', ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default='MTRCNN')
    args = parser.parse_args()

    model_name = args.model_name
    lr_init = config.lr_init
    epochs = config.epoch
    batch_size = config.batch_size

    sys_name = 'sys_' + model_name
    basic_name = sys_name + '_' + str(lr_init).replace('-', '')
    suffix, system_name = define_system_name(basic_name=basic_name, epochs=epochs, batch_size=batch_size)

    fold_num = 10
    cv_seed = 42
    clip_length = 1000

    origin_stdout = sys.stdout
    origin_stderr = sys.stderr

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    system_root = os.path.join(os.getcwd(), system_name)
    create_folder(system_root)
    summary_file = os.path.join(system_root, '10fold_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('')

    fold_results = []

    for fold_index in range(fold_num):
        fold_no = fold_index + 1
        system_path = os.path.join(system_root, 'fold_' + str(fold_no))

        models_dir = os.path.join(system_path, 'md')

        if suffix:
            models_dir = models_dir + '_' + suffix

        log_path = models_dir + '_log'

        if not os.path.exists(log_path):
            create_folder(log_path)
            filename = os.path.basename(__file__).split('.py')[0]

            if isinstance(sys.stdout, Logger):
                sys.stdout.close()
            if isinstance(sys.stderr, Logger):
                sys.stderr.close()

            sys.stdout = origin_stdout
            sys.stderr = origin_stderr

            print_log_file = os.path.join(log_path, filename + '_print.log')
            sys.stdout = Logger(print_log_file, origin_stdout)
            console_log_file = os.path.join(log_path, filename + '_console.log')
            sys.stderr = Logger(console_log_file, origin_stderr)

            fold_seed = cv_seed + fold_index
            print('==========================================')
            print('fold_no: ', fold_no)
            print('fold_seed: ', fold_seed)
            print('==========================================')

            set_seed(fold_seed)

            using_model = eval(model_name)
            model = using_model(10)
            model.to(device)

            generator = DataGeneratorEmotion(renormal=True, clip_length=clip_length, batch_size=batch_size,
                                             test_size=0.165, val_size=0.1, seed=cv_seed,
                                             fold_index=fold_index, folds_num=fold_num)

            fold_result = training_process_total_10fold(generator, model, models_dir, epochs, batch_size,
                                                        lr_init=lr_init, log_path=log_path,   device=device)
            fold_result['fold_no'] = fold_no
            fold_results.append(fold_result)

            fold_line = 'fold_no:  {0} best_val_acc_total: {1:.5f} final_val_acc_total: {2:.5f} best_val_acc_itera_total: {3:.3f}'.format(
                fold_no,
                float(fold_result['best_val_acc_total']),
                float(fold_result['final_val_acc_total']),
                float(fold_result['best_val_acc_itera_total']))
            print(fold_line)

    if isinstance(sys.stdout, Logger):
        sys.stdout.close()
    if isinstance(sys.stderr, Logger):
        sys.stderr.close()

    sys.stdout = origin_stdout
    sys.stderr = origin_stderr

    best_val_acc_total = [each['best_val_acc_total'] for each in fold_results]
    final_val_acc_total = [each['final_val_acc_total'] for each in fold_results]

    summary_lines = []
    summary_lines.append('================ 10 fold results ================')
    for each in fold_results:
        summary_lines.append('fold_no:  {0} best_val_acc_total: {1:.5f} final_val_acc_total: {2:.5f} best_val_acc_itera_total: {3:.3f}'.format(
            each['fold_no'],
            float(each['best_val_acc_total']),
            float(each['final_val_acc_total']),
            float(each['best_val_acc_itera_total'])))

    summary_lines.append('10 fold best_val_acc_total mean: %.5f std: %.5f'
                         % (float(np.mean(best_val_acc_total)), float(np.std(best_val_acc_total))))
    summary_lines.append('10 fold final_val_acc_total mean: %.5f std: %.5f'
                         % (float(np.mean(final_val_acc_total)), float(np.std(final_val_acc_total))))

    for each_line in summary_lines:
        print(each_line)

    with open(summary_file, 'a') as f:
        for each_line in summary_lines:
            f.write(each_line + '\n')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
