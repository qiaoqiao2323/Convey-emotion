import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder
from framework.models_pytorch import move_data_to_device
import framework.config as config
from sklearn import metrics
from framework.earlystop import EarlyStopping



def define_system_name(alpha=None, basic_name='system', att_dim=None, n_heads=None,
                       batch_size=None, epochs=None):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(batch_size) + '_e' + str(epochs) \
                 + '_attd' + str(att_dim) + '_h' + str(n_heads) if att_dim is not None and n_heads is not None \
        else '_b' + str(batch_size)  + '_e' + str(epochs)

    sys_suffix = sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name


def forward(model, generate_func, device, return_names = False):
    output = []
    label = []

    audio_names = []
    # Evaluate on mini-batch
    for num, data in enumerate(generate_func):
        (batch_x,  batch_y) = data

        batch_x = move_data_to_device(batch_x, device)

        model.eval()
        with torch.no_grad():
            output_linear = model(batch_x)

            output.append(output_linear.data.cpu().numpy())
            # ------------------------- labels -------------------------------------------------------------------------
            label.append(batch_y)

    dict = {}

    if return_names:
        dict['audio_names'] = np.concatenate(audio_names, axis=0)

    dict['output'] = np.concatenate(output, axis=0)
    # ----------------------------- labels -------------------------------------------------------------------------
    dict['label'] = np.concatenate(label, axis=0)
    return dict


def forward_total(model, generate_func, device, return_names = False):
    output_total = []
    label_total = []


    audio_names = []
    # Evaluate on mini-batch
    for num, data in enumerate(generate_func):
        (batch_x_a, batch_x_t,  batch_y_total) = data

        batch_x_a = move_data_to_device(batch_x_a, device)
        batch_x_t = move_data_to_device(batch_x_t, device)

        model.eval()
        with torch.no_grad():
            total = model(batch_x_a, batch_x_t)

            output_total.append(total.data.cpu().numpy())
            label_total.append(batch_y_total)

    dict = {}

    if return_names:
        dict['audio_names'] = np.concatenate(audio_names, axis=0)

    dict['output_total'] = np.concatenate(output_total, axis=0)
    dict['label_total'] = np.concatenate(label_total, axis=0)

    return dict



def cal_softmax_classification_accuracy(target, predict, average=None, eps=1e-8):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """
    # print(target)
    # print(predict)
    classes_num = predict.shape[-1]

    predict = np.argmax(predict, axis=-1)  # (audios_num,)
    samples_num = len(target)

    # print(classes_num)
    # print(predict)
    # print(samples_num)
    # # 3
    # # [1 0 1 1 0 0 0 2]
    # # 8

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    # print(correctness, total)
    # [0. 3. 1.] [0. 6. 2.]  ---- 这样的话，因为第0类没有，0/0会有nan, 所以加 eps, 0/0....8
    accuracy = correctness / (total + eps)

    if average == 'each_class':
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def evaluate(model, generate_func, device):
    # Forward
    dict = forward(model=model, generate_func=generate_func, device=device)

    # softmax classification acc
    acc = cal_softmax_classification_accuracy(dict['label'], dict['output'], average = 'macro')

    return acc



def evaluate_total(model, generate_func, device):
    # Forward
    dict = forward_total(model=model, generate_func=generate_func, device=device)

    # softmax clas
    acc_total = cal_softmax_classification_accuracy(dict['label_total'], dict['output_total'], average='macro')

    return acc_total


def training_process(generator, model, models_dir, epochs, batch_size, lr_init=1e-3,
                                  log_path=None, device=None):
    create_folder(models_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    max_testing_acc = 0.000001
    max_testing_acc_itera = 0
    save_testing_best = 0
    list_testing_acc = []
    list_testing_acc_file = os.path.join(log_path, 'testing_acc.txt')

    # ------------------------------------------------------------------------------------------------------------------

    sample_num = len(generator.training_ids)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = int(one_epoch )
    print('validating every: ', check_iter, ' iteration')


    training_start_time = time.time()
    for iteration, all_data in enumerate(generator.generate_training()):
        (batch_x, batch_y) = all_data

        batch_x = move_data_to_device(batch_x, device)
        batch_y = move_data_to_device(batch_y, device)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        gesture_linear = model(batch_x)

        x_softmax = F.log_softmax(gesture_linear, dim=-1)
        loss = F.nll_loss(x_softmax, batch_y)

        loss.backward()
        optimizer.step()

        Epoch = iteration / one_epoch

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss))

        # 6122 / 64 = 95.656
        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()
            # Generate function
            generate_func = generator.generate_validate(data_type='validate')
            val_acc = evaluate(model=model, generate_func=generate_func, device=device)

            print('E: ', '%.3f' % (Epoch), 'val_acc: %.3f' % float(val_acc))

            train_time = train_fin_time - train_bgn_time

            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))
            #------------------------ validation done ------------------------------------------------------------------


            #-------------------------each epoch testing----------------------------------------------------------------
            print('----------------------evaluating--------------------------------')
            generate_func = generator.generate_testing(data_type='testing')
            test_acc = evaluate(model=model, generate_func=generate_func, device=device)
            testing_time = time.time() - train_fin_time

            list_testing_acc.append(test_acc)
            if test_acc > max_testing_acc:
                max_testing_acc = test_acc
                save_testing_best = 1
                max_testing_acc_itera = Epoch

            print('E: ', '%.3f' % (Epoch), 'test_acc: %.3f' % float(test_acc))

            print('E: {}, T_testing: {:.3f} s,  '   
                  'max_testing_acc: {:.3f} , itera: {}, '
                  .format('%.4f' % (Epoch),
                          testing_time,
                          max_testing_acc, max_testing_acc_itera))

            np.savetxt(list_testing_acc_file, list_testing_acc,  fmt='%.5f')

            if save_testing_best:
                save_testing_best = 0
                # save_out_dict = {'iteration': iteration, 'state_dict': model.state_dict(),
                #                  'optimizer': optimizer.state_dict()}
                save_out_dict = model.state_dict()
                save_out_path = os.path.join(models_dir, 'best_testing' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best scene model saved to {}'.format(save_out_path))

        # Stop learning
        if iteration > (epochs * one_epoch):
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            # correct
            save_out_dict = model.state_dict()
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))

            print('Training is done!!!')

            print('E: {}, max_testing_acc: {:.3f} , itera: {}, ' .format('%.4f' % (Epoch),
                                                                         max_testing_acc, max_testing_acc_itera))

            np.savetxt(list_testing_acc_file, list_testing_acc, fmt='%.5f')
            print('Training is done!!!')

            print('----------------------evaluating--------------------------------')
            generate_func = generator.generate_testing(data_type='testing')
            test_acc = evaluate(model=model, generate_func=generate_func, device=device)

            print('E: ', '%.3f' % (Epoch),  'test_acc: %.3f' % float(test_acc))
            break


def training_process_10fold(generator, model, models_dir, epochs, batch_size, lr_init=1e-3,
                            log_path=None, device=None):
    create_folder(models_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    max_val_acc = 0.000001
    max_val_acc_itera = 0
    save_val_best = 0
    list_val_acc = []
    list_val_acc_file = os.path.join(log_path, 'val_acc.txt')

    sample_num = len(generator.training_ids)
    one_epoch = max(1, int(sample_num / batch_size))
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = max(1, int(one_epoch ))
    print('validating every: ', check_iter, ' iteration')

    final_val_acc = 0
    train_fin_time = time.time()

    training_start_time = time.time()
    for iteration, all_data in enumerate(generator.generate_training()):
        (batch_x, batch_y) = all_data

        batch_x = move_data_to_device(batch_x, device)
        batch_y = move_data_to_device(batch_y, device)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        output = model(batch_x)

        x_softmax = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(x_softmax, batch_y)

        loss.backward()
        optimizer.step()

        Epoch = iteration / one_epoch

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss.detach()))

        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()

            generate_func = generator.generate_validate(data_type='validate')
            val_acc = evaluate(model=model, generate_func=generate_func, device=device)
            list_val_acc.append(val_acc)

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                save_val_best = 1
                max_val_acc_itera = Epoch

            print('E: ', '%.3f' % (Epoch), 'val_acc: %.3f' % float(val_acc))

            train_time = train_fin_time - train_bgn_time

            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))

            print('E: {}, T_validate: {:.3f} s,  '
                  ' max_val_acc: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch),
                          validate_time,
                          max_val_acc, max_val_acc_itera))

            np.savetxt(list_val_acc_file, list_val_acc, fmt='%.5f')

            if save_val_best:
                save_val_best = 0
                save_out_dict = model.state_dict()
                save_out_path = os.path.join(models_dir, 'best_validation' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best scene model saved to {}'.format(save_out_path))

        if iteration > (epochs * one_epoch):
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            save_out_dict = model.state_dict()
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Training is done!!!')

            print('----------------------evaluating--------------------------------')
            generate_func = generator.generate_validate(data_type='validate')
            final_val_acc = evaluate(model=model, generate_func=generate_func, device=device)
            list_val_acc.append(final_val_acc)

            if final_val_acc > max_val_acc:
                max_val_acc = final_val_acc
                max_val_acc_itera = Epoch

            print('E: ', '%.3f' % (Epoch),
                  'final_val_acc: %.3f' % float(final_val_acc))

            print('E: {}, T_validate: {:.3f} s,  '
                  ' max_val_acc: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch),
                          time.time() - train_fin_time,
                          max_val_acc, max_val_acc_itera))

            np.savetxt(list_val_acc_file, list_val_acc, fmt='%.5f')
            break

    return {'best_val_acc': max_val_acc,
            'best_val_acc_itera': max_val_acc_itera,
            'final_val_acc': final_val_acc}


def training_process_total(generator, model, models_dir, epochs, batch_size, lr_init=1e-3,
                                  log_path=None, device=None):
    create_folder(models_dir)

    # # Optimizer
    # if adamw:
    #     optimizer = optim.AdamW(model.parameters(), lr=lr_init)
    # else:
    optimizer = optim.Adam(model.parameters(), lr=lr_init)

    max_testing_acc_total = 0.000001
    max_testing_acc_itera_total = 0
    save_testing_best_total = 0
    list_testing_acc_total = []
    list_testing_acc_file_total = os.path.join(log_path, 'testing_acc_total.txt')

    list_val_acc_total = []
    list_val_acc_file_total = os.path.join(log_path, 'val_acc_total.txt')

    # ------------------------------------------------------------------------------------------------------------------

    sample_num = len(generator.training_ids)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = int(one_epoch )
    print('validating every: ', check_iter, ' iteration')


    training_start_time = time.time()
    for iteration, all_data in enumerate(generator.generate_training()):
        (batch_x_a, batch_x_t, batch_y_total) = all_data

        batch_x_a = move_data_to_device(batch_x_a, device)
        batch_x_t = move_data_to_device(batch_x_t, device)
        batch_y_total = move_data_to_device(batch_y_total, device)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        total = model(batch_x_a, batch_x_t)

        total_softmax = F.log_softmax(total, dim=-1)
        # print(total_softmax.shape) # torch.Size([32, 10])
        # print(batch_y_total.shape) # torch.Size([32])

        loss = F.nll_loss(total_softmax, batch_y_total)

        loss.backward()
        optimizer.step()

        Epoch = iteration / one_epoch

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss))

        # 6122 / 64 = 95.656
        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()
            # Generate function
            generate_func = generator.generate_validate(data_type = 'validate')
            val_acc_total = evaluate_total(model=model, generate_func=generate_func, device=device)
            list_val_acc_total.append(val_acc_total)

            print('E: ', '%.3f' % (Epoch),
                  'val_acc_total: %.3f' % float(val_acc_total) )

            train_time = train_fin_time - train_bgn_time

            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))
            #------------------------ validation done ------------------------------------------------------------------


            #-------------------------each epoch testing----------------------------------------------------------------
            print('----------------------evaluating--------------------------------')
            generate_func = generator.generate_testing(data_type='testing')
            test_acc_total = evaluate_total(model=model, generate_func=generate_func, device=device)
            list_testing_acc_total.append(test_acc_total)

            testing_time = time.time() - train_fin_time
            if test_acc_total > max_testing_acc_total:
                max_testing_acc_total = test_acc_total
                save_testing_best_total = 1
                max_testing_acc_itera_total = Epoch

            print('E: ', '%.3f' % (Epoch),
                   'test_acc_total: %.3f' % float(test_acc_total))

            print('E: {}, T_testing: {:.3f} s,  '   
                  ' max_testing_acc_total: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch),
                          testing_time,
                          max_testing_acc_total, max_testing_acc_itera_total))

            np.savetxt(list_testing_acc_file_total, list_testing_acc_total, fmt='%.5f')
            np.savetxt(list_val_acc_file_total, list_val_acc_total, fmt='%.5f')


            if save_testing_best_total:
                save_testing_best_total = 0
                save_out_dict = model.state_dict()
                save_out_path = os.path.join(models_dir, 'best_testing_total' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best scene model saved to {}'.format(save_out_path))

        # Stop learning
        if iteration > (epochs * one_epoch):
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            # correct
            save_out_dict = model.state_dict()
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))

            print('Training is done!!!')

            print('E: ', '%.3f' % (Epoch),
                  'test_acc_total: %.3f' % float(test_acc_total))

            print('E: {}, T_testing: {:.3f} s,  '
                  ' max_testing_acc_total: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch),
                          testing_time,
                          max_testing_acc_total, max_testing_acc_itera_total))

            np.savetxt(list_testing_acc_file_total, list_testing_acc_total, fmt='%.5f')

            print('Training is done!!!')

            print('----------------------evaluating--------------------------------')
            # 这里多出 0.05轮的模型，才是最终保存的模型

            generate_func = generator.generate_validate(data_type='validate')
            val_acc = evaluate_total(model=model, generate_func=generate_func, device=device)
            list_val_acc_total.append(val_acc)

            generate_func = generator.generate_testing(data_type='testing')
            test_acc = evaluate_total(model=model, generate_func=generate_func, device=device)
            list_testing_acc_total.append(test_acc)

            np.savetxt(list_testing_acc_file_total, list_testing_acc_total, fmt='%.5f')
            np.savetxt(list_val_acc_file_total, list_val_acc_total, fmt='%.5f')
            break



def training_process_total_10fold(generator, model, models_dir, epochs, batch_size, lr_init=1e-3,
                                  log_path=None, device=None):
    create_folder(models_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr_init)

    max_val_acc_total = 0.000001
    max_val_acc_itera_total = 0
    save_val_best_total = 0
    list_val_acc_total = []
    list_val_acc_file_total = os.path.join(log_path, 'val_acc_total.txt')

    sample_num = len(generator.training_ids)
    one_epoch = max(1, int(sample_num / batch_size))
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = max(1, int(one_epoch ))
    print('validating every: ', check_iter, ' iteration')

    final_val_acc_total = 0
    train_fin_time = time.time()

    training_start_time = time.time()
    for iteration, all_data in enumerate(generator.generate_training()):
        (batch_x_a, batch_x_t, batch_y_total) = all_data

        batch_x_a = move_data_to_device(batch_x_a, device)
        batch_x_t = move_data_to_device(batch_x_t, device)
        batch_y_total = move_data_to_device(batch_y_total, device)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        total = model(batch_x_a, batch_x_t)

        total_softmax = F.log_softmax(total, dim=-1)
        loss = F.nll_loss(total_softmax, batch_y_total)

        loss.backward()
        optimizer.step()

        Epoch = iteration / one_epoch

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss.detach()))

        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()

            generate_func = generator.generate_validate(data_type='validate')
            val_acc_total = evaluate_total(model=model, generate_func=generate_func, device=device)
            list_val_acc_total.append(val_acc_total)

            if val_acc_total > max_val_acc_total:
                max_val_acc_total = val_acc_total
                save_val_best_total = 1
                max_val_acc_itera_total = Epoch

            print('E: ', '%.3f' % (Epoch),
                  'val_acc_total: %.3f' % float(val_acc_total))

            train_time = train_fin_time - train_bgn_time

            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))

            print('E: {}, T_validate: {:.3f} s,  '
                  ' max_val_acc_total: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch),
                          validate_time,
                          max_val_acc_total, max_val_acc_itera_total))

            np.savetxt(list_val_acc_file_total, list_val_acc_total, fmt='%.5f')

            if save_val_best_total:
                save_val_best_total = 0
                save_out_dict = model.state_dict()
                save_out_path = os.path.join(models_dir, 'best_validation_total' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best scene model saved to {}'.format(save_out_path))

        if iteration > (epochs * one_epoch):
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            save_out_dict = model.state_dict()
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Training is done!!!')

            print('----------------------evaluating--------------------------------')
            generate_func = generator.generate_validate(data_type='validate')
            final_val_acc_total = evaluate_total(model=model, generate_func=generate_func, device=device)
            list_val_acc_total.append(final_val_acc_total)

            if final_val_acc_total > max_val_acc_total:
                max_val_acc_total = final_val_acc_total
                max_val_acc_itera_total = Epoch

            print('E: ', '%.3f' % (Epoch),
                  'final_val_acc_total: %.3f' % float(final_val_acc_total))

            print('E: {}, T_validate: {:.3f} s,  '
                  ' max_val_acc_total: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch),
                          time.time() - train_fin_time,
                          max_val_acc_total, max_val_acc_itera_total))

            np.savetxt(list_val_acc_file_total, list_val_acc_total, fmt='%.5f')
            break

    return {'best_val_acc_total': max_val_acc_total,
            'best_val_acc_itera_total': max_val_acc_itera_total,
            'final_val_acc_total': final_val_acc_total}

