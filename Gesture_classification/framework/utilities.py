import os
import numpy as np

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    # 实际计算：直接变形，拉成64行，每一行一个结果
    # stdev = np.std(data.reshape(-1, 1), axis=0)
    #     print("Standard Deviation: ", stdev)
    #     # Standard Deviation:  (64,)
    #     print("Standard Deviation: ", stdev.shape)
    #     # 与上面的结果一致，看来就是这么算的

    # mean = np.mean(data, axis=0)
    #     mean = np.mean(mean, axis=0)
    #     print("Mean: ", mean)
    #     # Mean:  [-39.047832]  与上面结果一致
    #     print("Mean: ", mean.shape)
    #     # Mean:  (64,)  与上面结果一致

    return mean, std


def scale(x, mean, std):
    return (x - mean) / std


# ------------------ demo ---------------------------------------------------------------------------------------------
def calculate_scalar_demo(x):
    # print(x)
    # print(x.shape)
    assert True not in np.isnan(x)

    # for each in range(len(x)):
    #     row = x[each]
    #     # print(row)
    #     # print(row.shape)
    #     assert True not in np.isnan(row)

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    # print('mean: ', mean)
    # [ -6.018424   -6.551033   -6.8909006  -7.135826   -7.3510823  -7.480673
    #   -7.571218   -7.680842   -7.764014   -7.839708   -7.9108295  -7.969385
    #   -7.9846487  -8.029721   -8.048612   -8.083075   -8.156022   -8.228257
    #   -8.359451   -8.483957   -8.639418   -8.791793   -8.884309   -8.9201565
    #   -8.938811   -8.966767   -9.034953   -9.119106   -9.218562   -9.32662
    #   -9.430841   -9.535318   -9.650718   -9.772229   -9.883647   -9.913938
    #   -9.935442  -10.049141  -10.1867075 -10.362941  -10.548183  -10.69776
    #  -10.799923  -10.865824  -10.9274645 -10.995023  -11.070687  -11.15063
    #  -11.247276  -11.373701  -11.491775  -11.61711   -12.376908  -13.8216095
    #  -13.986309  -14.037714  -14.084533  -14.127888  -14.167356  -14.20305
    #  -14.233981  -14.260572  -14.281123  -14.294628 ]
    # print('std: ', std)
    # [0.9275274  0.9503843  0.9411956  0.9516942  0.9656475  0.98818105
    #  1.0112762  1.0218291  1.0353186  1.0504278  1.0469218  1.043648
    #  1.0442965  1.0435324  1.0453404  1.0440447  1.0421628  1.0432991
    #  1.0396047  1.0428373  1.0502497  1.055006   1.0665127  1.0781437
    #  1.0783657  1.0773524  1.080429   1.0851995  1.091652   1.0976961
    #  1.1062087  1.1076928  1.1024603  1.1059903  1.1096089  1.1136392
    #  1.1156033  1.1243112  1.125575   1.126428   1.1284149  1.130232
    #  1.1339173  1.1415312  1.1537234  1.1823567  1.1952301  1.2102163
    #  1.2435135  1.2545516  1.2635572  1.313806   1.2335336  1.2045804
    #  1.2801212  1.2843628  1.2872167  1.2894176  1.2912995  1.2930605
    #  1.294887   1.296844   1.2990457  1.3014892 ]
    # print('mean: ', mean.shape)
    # print('std: ', std.shape)
    # mean:  (64,)
    # std:  (64,)

    # data = x
    # # mean = sum(data) / len(data)
    # mean = np.mean(data, axis=0)
    # # print("Mean: ", mean.shape)
    # # # Mean:  (480, 64)
    # mean = np.mean(mean, axis=0)
    # # print("Mean: ", mean.shape)
    # # Mean:  (64,)  与上面结果一致
    #
    # stdev = np.std(data.reshape(-1, 64), axis=0)
    # # print("Standard Deviation: ", stdev)
    # # # Standard Deviation:  (64,)
    # # print("Standard Deviation: ", stdev.shape)
    # # # 与上面的结果一致，看来就是这么算的

    # assert True not in np.isnan(mean)
    # # assert True not in np.isnan(std)
    #
    # assert True not in np.isnan(stdev)

    return mean, std


def calculate_scalar_dbFS_RMS(x):
    # print(x)
    # print(x.shape)
    # assert True not in np.isnan(x)
    #
    # for each in range(len(x)):
    #     row = x[each]
    #     # print(row)
    #     # print(row.shape)
    #     assert True not in np.isnan(row)

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    print('numpy mean: ', mean)
    # [-39.047832]

    data = x
    # print("sum(data): ", sum(data))
    # mean = sum(data) / len(data)
    # mean = np.mean(data, axis=0)
    # print("Mean: ", mean.shape)
    # # Mean:  (480, 64)
    mean = np.mean(data, axis=0)
    mean = np.mean(mean, axis=0)
    print("Mean: ", mean)
    # Mean:  [-39.047832]  与上面结果一致
    print("Mean: ", mean.shape)
    # Mean:  (64,)  与上面结果一致

    print(x.shape)

    jjjj

    print('x[0]:', x[0])
    print('sum x[0]:', sum(x[0]))

    print('x[:, 0]:', x[:, 0])

    for each in range(len(x[:, 0])):
        row = x[:, 0][each]
        print("row: ", each, row)
        # print(row.shape)
        assert not np.isnan(row)

    print('sum x[:, 0]:', sum(x[:, 0]))
    kkkk

    stdev = np.std(data.reshape(-1, 1), axis=0)
    print("Standard Deviation: ", stdev)
    # Standard Deviation:  (64,)
    print("Standard Deviation: ", stdev.shape)
    # 与上面的结果一致，看来就是这么算的

    assert True not in np.isnan(mean)
    # assert True not in np.isnan(std)

    assert True not in np.isnan(stdev)

    return mean, std

# ----------------------------------------------------------------------------------------------------------------------

from sklearn import metrics
def cal_acc_auc(predictions, targets):
    tagging_truth_label_matrix = targets
    pre_tagging_label_matrix = predictions

    # overall
    tp = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix > 1.5)
    fn = np.sum(tagging_truth_label_matrix - pre_tagging_label_matrix > 0.5)
    fp = np.sum(pre_tagging_label_matrix - tagging_truth_label_matrix > 0.5)
    tn = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix < 0.5)

    Acc = (tp + tn) / (tp + tn + fp + fn)

    aucs = []
    for i in range(targets.shape[0]):
        test_y_auc, pred_auc = targets[i, :], predictions[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc = sum(aucs) / len(aucs)
    return Acc, final_auc





