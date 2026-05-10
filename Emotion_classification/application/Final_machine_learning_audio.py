import random
import sys, os, argparse
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.utilities import create_folder
from framework.data_generator import DataGeneratorEmotion


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


def get_search_params(model_name):
    if model_name == 'RF':
        params = [{"max_features": "sqrt"},
                  {"max_features": "log2"},
                  {"max_features": None}]
    elif model_name == 'SVM':
        params = [{"C": 1.0}]
    elif model_name == 'DT':
        params = [{"ccp_alpha": 0.0}]
    else:
        raise ValueError('Unknown model_name: ' + model_name)

    return params


def build_model(model_name, params):
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=100, max_features=params["max_features"])
    elif model_name == 'SVM':
        model = LinearSVC(random_state=100, max_iter=10000, C=params["C"])
    elif model_name == 'DT':
        model = DecisionTreeClassifier(random_state=100, ccp_alpha=params["ccp_alpha"])
    else:
        raise ValueError('Unknown model_name: ' + model_name)

    return model


def get_ml_features(feature):
    feature = feature.reshape((feature.shape[0], -1))

    return feature


def main(argv):
    model_name = ['RF', 'SVM', 'DT']

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default='RF')
    args = parser.parse_args()

    model_name = args.model_name

    system_name = 'sys_ML_' + model_name
    system_path = os.path.join(os.getcwd(), system_name)
    create_folder(system_path)

    log_path = os.path.join(system_path, 'log')

    if not os.path.exists(log_path):
        create_folder(log_path)

        filename = os.path.basename(__file__).split('.py')[0]
        print_log_file = os.path.join(log_path, filename + '_print.log')
        console_log_file = os.path.join(log_path, filename + '_console.log')
        origin_stdout = sys.stdout
        origin_stderr = sys.stderr
        sys.stdout = Logger(print_log_file, origin_stdout)
        sys.stderr = Logger(console_log_file, origin_stderr)

        summary_file = os.path.join(system_path, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write('')

        set_seed(42)
        generator = DataGeneratorEmotion(renormal=True, clip_length=1000, batch_size=32,
                                         test_size=0.165, val_size=0.1, seed=42, modality='audio')

        train_feature = generator.transform(generator.training_feature, generator.mean_feature, generator.std_feature)
        val_feature = generator.transform(generator.validation_feature, generator.mean_feature, generator.std_feature)
        test_feature = generator.transform(generator.test_feature, generator.mean_feature, generator.std_feature)

        X_train = get_ml_features(train_feature)
        X_val = get_ml_features(val_feature)
        X_test = get_ml_features(test_feature)
        y_train = generator.training_label
        y_val = generator.validation_label
        y_test = generator.test_label

        best_val_acc = -1
        best_params = None
        final_model = None

        for each_params in get_search_params(model_name):
            model = build_model(model_name, each_params)
            model.fit(X_train, y_train)

            val_predictions = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_predictions)
            print('val_acc: %.5f best_params: %s' % (float(val_acc), str(each_params)))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = each_params
                final_model = model

        test_predictions = final_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_predictions)
        cm_labels = sorted(set(y_test))
        cm = confusion_matrix(y_test, test_predictions, labels=cm_labels)
        report = classification_report(y_test, test_predictions, labels=cm_labels, zero_division=0)

        summary_lines = []
        summary_lines.append('================ validation results ================')
        summary_lines.append('best_val_acc: %.5f' % float(best_val_acc))
        summary_lines.append('best_params: ' + str(best_params))
        summary_lines.append('================ test results ================')
        summary_lines.append('test_acc: %.5f' % float(test_acc))
        summary_lines.append('confusion_matrix_labels: ' + str(cm_labels))
        summary_lines.append(str(cm))
        summary_lines.append(report)

        for each_line in summary_lines:
            print(each_line)

        with open(summary_file, 'a') as f:
            for each_line in summary_lines:
                f.write(each_line + '\n')

        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = origin_stdout
        sys.stderr = origin_stderr


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
