import random
import sys, os, argparse


# # 这里的0是GPU id
# gpu_id = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
#
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

    def flush(self):
        pass



def main(argv):
    model_name = ['CNN_LSTM', 'CNN_Transformer', 'CNN_GRU', 'MTRCNN', ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default='MTRCNN')
    args = parser.parse_args()

    model_name = args.model_name
    lr_init = config.lr_init
    epochs = config.epoch
    batch_size = config.batch_size

    sys_name = 'sys_' + model_name + '_audio'
    basic_name = sys_name + '_' + str(lr_init).replace('-', '')
    suffix, system_name = define_system_name(basic_name=basic_name, epochs=epochs, batch_size=batch_size)

    system_path = os.path.join(os.getcwd(), system_name)

    models_dir = os.path.join(system_path, 'md') + '_' + suffix

    log_path = models_dir + '_log'

    if not os.path.exists(log_path):
        create_folder(log_path)
        filename = os.path.basename(__file__).split('.py')[0]
        print_log_file = os.path.join(log_path, filename + '_print.log')
        sys.stdout = Logger(print_log_file, sys.stdout)
        console_log_file = os.path.join(log_path, filename + '_console.log')
        sys.stderr = Logger(console_log_file, sys.stderr)

        using_model = eval(model_name)

        model = SingleModalModel(using_model(len(config.Gesture_list_name)), 'audio')

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        model.to(device)

        clip_length = 1000
        generator = DataGeneratorgesture(renormal=True, clip_length=clip_length, batch_size=batch_size,
                                         test_size=0.165, val_size=0.1, modality='audio')

        training_process(generator, model, models_dir, epochs, batch_size, lr_init=lr_init,
                         log_path=log_path,  device=device)
        print('Training is done!!!')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















