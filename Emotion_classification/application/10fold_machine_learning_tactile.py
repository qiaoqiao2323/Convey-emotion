import random
import sys, os, argparse
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
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


def build_model(model_name):
    cv_10 = KFold(n_splits=10, shuffle=True, random_state=100)

    if model_name == 'RF':
        model = GridSearchCV(
            estimator=RandomForestClassifier(random_state=100),
            param_grid={"max_features": ["sqrt", "log2", None]},
            cv=cv_10,
            scoring="accuracy",
            n_jobs=-1
        )
    elif model_name == 'SVM':
        model = GridSearchCV(
            estimator=LinearSVC(random_state=100, max_iter=10000),
            param_grid={"C": [1.0]},
            cv=cv_10,
            scoring="accuracy",
            n_jobs=-1
        )
    elif model_name == 'DT':
        model = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=100),
            param_grid={"ccp_alpha": [0.0]},
            cv=cv_10,
            scoring="accuracy",
            n_jobs=-1
        )
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

    system_name = 'sys_CV_ML_' + model_name
    system_path = os.path.join(os.getcwd(), system_name)
    create_folder(system_path)

    log_path = os.path.join(system_path, 'log')
    create_folder(log_path)

    filename = os.path.basename(__file__).split('.py')[0]
    print_log_file = os.path.join(log_path, filename + '_print.log')
    console_log_file = os.path.join(log_path, filename + '_console.log')
    origin_stdout = sys.stdout
    origin_stderr = sys.stderr
    sys.stdout = Logger(print_log_file, origin_stdout)
    sys.stderr = Logger(console_log_file, origin_stderr)

    summary_file = os.path.join(system_path, '10fold_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('')

    set_seed(42)

    fold_results = []

    for fold_index in range(10):
        fold_no = fold_index + 1
        generator = DataGeneratorEmotion(renormal=True, clip_length=1000, batch_size=32,
                                         test_size=0.165, val_size=0.1, seed=42,
                                         fold_index=fold_index, folds_num=10, modality='tactile')

        train_feature = generator.transform(generator.training_feature, generator.mean_feature, generator.std_feature)
        val_feature = generator.transform(generator.validation_feature, generator.mean_feature, generator.std_feature)

        X_train = get_ml_features(train_feature)
        X_val = get_ml_features(val_feature)
        y_train = generator.training_label
        y_val = generator.validation_label

        model = build_model(model_name)
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_predictions)

        fold_result = {'fold_no': fold_no,
                       'val_acc': val_acc,
                       'best_params': model.best_params_}
        fold_results.append(fold_result)

        fold_line = 'fold_no:  {0} val_acc: {1:.5f} best_params: {2}'.format(
            fold_no, float(val_acc), model.best_params_)
        print(fold_line)

    val_acc_list = [each['val_acc'] for each in fold_results]

    generator = DataGeneratorEmotion(renormal=True, clip_length=1000, batch_size=32,
                                     test_size=0.165, val_size=0.1, seed=42,
                                     use_all_training_ids=True, modality='tactile')

    train_feature = generator.transform(generator.training_feature, generator.mean_feature, generator.std_feature)
    test_feature = generator.transform(generator.test_feature, generator.mean_feature, generator.std_feature)

    X_train_total = get_ml_features(train_feature)
    X_test = get_ml_features(test_feature)
    y_train_total = generator.training_label
    y_test = generator.test_label

    final_model = build_model(model_name)
    final_model.fit(X_train_total, y_train_total)
    test_predictions = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_predictions)
    cm_labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, test_predictions, labels=cm_labels)
    report = classification_report(y_test, test_predictions, labels=cm_labels, zero_division=0)

    summary_lines = []
    summary_lines.append('================ 10 fold results ================')
    for each in fold_results:
        summary_lines.append('fold_no:  {0} val_acc: {1:.5f} best_params: {2}'.format(
            each['fold_no'], float(each['val_acc']), each['best_params']))
    summary_lines.append('10 fold val_acc mean: %.5f std: %.5f'
                         % (float(np.mean(val_acc_list)), float(np.std(val_acc_list))))
    summary_lines.append('================ true test results ================')
    summary_lines.append('test_acc: %.5f' % float(test_acc))
    summary_lines.append('best_params_on_all_training_ids: ' + str(final_model.best_params_))
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
