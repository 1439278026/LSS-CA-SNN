import torch
import torch.nn as nn
from tools import functional, surrogate
from tools.CUPY_network import *
from tools.datasets import EEGLoader, EEGLoader2
from tools.utils import seed_torch, EarlyStopping
import argparse
import numpy as np
import random
import os
from tqdm import trange, tqdm
import time
import logging




# Initialize the parser and logging
parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network')
parser.add_argument('--dataset', type=int, default=0, help='Choose Dataset')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--trial_num', type=int, default=5, help='Number of repeated experiments')
parser.add_argument('--seed', type=int, default=2024, help='whether EA')
parser.add_argument('--loo', type=bool, default=False, help='cross sessions')
parser.add_argument('--patience', type=int, default=30, help='Early Stop Tolerance')
parser.add_argument('--model', type=str, default='SpikeCNN2D', help='Choose the model to train')
parser.add_argument('--device', type=str, default='0', help='Choose GPU')
parser.add_argument('--prep', type=str, default='/spike100', help='Choose form of spike data')
prms = vars(parser.parse_args())



select_dataset = {0: '/data', 1: '/data1'}
session_list = {0: range(14), 1: range(12)}
class_list = {0: 3, 1: 3}
inchannel_list = {0: 80, 1: 66}
T_list = {'/spike': 12001, '/spike1': 500, '/spike2': 250, '/feature': 8,
          '/spike100': 100, '/spike150': 150, '/spike200': 200, '/spike50': 50
          }
spiltratio = [0.2, 0.1, 0.7]  

ANN_dict = {'EEGNet': EEGNet, 'ShallowConvNet': ShallowConvNet, 'Conformer': Conformer,
            'DeepConvNet': DeepConvNet}
SNN_dict = {'LSS_CA_SNN': LSS_CA_SNN,}




if __name__ == '__main__':

#     # Configure logging
    if prms['loo']:
        log_dir = "/data2/fht/invasive/crossval/"
    else:
        log_dir = '/data2/fht/invasive/finaltest/'
    name = '_best-lca_'
    filename = f"dataset{prms['dataset']}_len{T_list[prms['prep']]}_{prms['model']}_loo{prms['loo']}{name}"
    log_filename = log_dir + filename + f".log"
    print(log_filename)
    logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.info("Starting training with parameters: %s", prms)

    model, in_channel_num, class_num, T, device = prms['model'], inchannel_list[prms['dataset']], class_list[
        prms['dataset']], T_list[prms['prep']], prms['device']
    device = 'cuda:' + str(device)
    loo = prms['loo']
    pick_session = session_list[prms['dataset']]
    workpath = '/data2/fht/invasive'
    if T <= 200:
        dirs = '/data2/fht/invasive'
    else:
        dirs = '/data2/syang/IBCI'
    data_path = dirs + select_dataset[prms['dataset']] + prms['prep']
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    print(prms, data_path)
    logging.info("Data path: %s", data_path)
    patience = prms['patience']
    trial_acc = []
    start_time = time.time()

    for trial_num in range(prms['trial_num']):  # replication
        seedval = prms['seed'] + trial_num
        seed_torch(seedval)
        session_test_acc = []
        logging.info("---------Starting trial number: %d", trial_num)

        for session in pick_session:
            logging.info("Training session: %d", session)
            if loo:
                model_path = workpath + '/model' + select_dataset[prms['dataset']] + prms[
                    'prep'] + '/' + model + name + '_loo_{}_{}.pth'.format(session, seedval)
            else:
                model_path = workpath + '/model' + select_dataset[prms['dataset']] + prms[
                    'prep'] + '/' + model + name + '_{}_{}.pth'.format(session, seedval)

            if model in ANN_dict:
                net = ANN_dict[model](in_channels=in_channel_num, time_step=T, classes_num=class_num).cuda()
            elif model in SNN_dict:
                net = SNN_dict[model](in_channels=in_channel_num, out_num=class_num, w=0.5,
                                      surrogate_function=surrogate.Sigmoid(), time_step=T, neuron='PLIF').cuda()

            if trial_num == 0 and session == 0:
                logging.info("Model architecture: %s", net)

            if not os.path.isdir(workpath + '/model' + select_dataset[prms['dataset']] + prms['prep']):
                os.mkdir(workpath + '/model' + select_dataset[prms['dataset']] + prms['prep'])

            logging.info("Model saved at: %s", model_path)
            optimizer = torch.optim.Adam(net.parameters(), lr=prms['lr'])
            loss_function = nn.CrossEntropyLoss().cuda()

            if loo:
                train_set = EEGLoader2(path=data_path, pick_session=(session,), all_session=pick_session,
                                       settup='train', spiltratio=spiltratio)
                validate_set = EEGLoader2(path=data_path, pick_session=(session,), all_session=pick_session,
                                          settup='validate', spiltratio=spiltratio)
                test_set = EEGLoader2(path=data_path, pick_session=(session,), all_session=pick_session, settup='test',
                                      spiltratio=spiltratio)
            else:
                train_set = EEGLoader(path=data_path, pick_session=session, settup='train', spiltratio=spiltratio)
                validate_set = EEGLoader(path=data_path, pick_session=session, settup='validate', spiltratio=spiltratio)
                test_set = EEGLoader(path=data_path, pick_session=session, settup='test', spiltratio=spiltratio)

            logging.info("Training set size: %d, Validation set size: %d, Test set size: %d", train_set.__len__(),
                         validate_set.__len__(), test_set.__len__())

            train_data_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=prms['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=0
            )
            validate_data_loader = torch.utils.data.DataLoader(
                dataset=validate_set,
                batch_size=1,
                shuffle=False,
                drop_last=False
            )
            test_data_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1,
                shuffle=False,
                drop_last=False
            )

            EarlyStop = EarlyStopping(patience=prms['patience'], path=model_path)
            train_acc = []
            train_loss = []
            val_acc = []
            val_loss = []
            test_acc_epoch = []
            descriptions = ['Train', 'Val']
            progress_bars = [tqdm(total=prms['epoch'], desc=desc, position=i, leave=False) for i, desc in
                             enumerate(descriptions)]

            for epoch in range(prms['epoch']):
                net.train()
                accuracy = 0
                loss0 = 0
                train_num = 0
                num = 0

                for frame, label in train_data_loader:  
                    frame = frame.cuda()  
                    label = label.reshape(-1).cuda()
                    out_fr = net(frame)  # N,num_classes
                    loss = loss_function(out_fr, label)
                    loss0 += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    accuracy += (out_fr.argmax(dim=1) == label.cuda()).float().sum().item()
                    train_num += label.numel() 
                    num += 1
                    progress_bars[0].set_postfix(loss='%.4f' % loss, epoch=epoch, acc=accuracy / train_num,
                                                 session=session, num=trial_num)
                    functional.reset_net(net)

                accuracy /= train_num
                loss0 /= num
                train_acc.append(accuracy)
                train_loss.append(loss0)
                logging.info(f"Epoch {epoch}, Training Accuracy: {accuracy:.4f}, Loss: {loss0:.4f}")

                net.eval()
                val_accuracy = 0
                loss0 = 0
                val_num = 0
                num = 0

                with torch.no_grad():
                    for frame, label in validate_data_loader:
                        frame = frame.cuda()  
                        label = label.reshape(-1).cuda()
                        out_fr = net(frame)  # N,num_classes
                        loss = loss_function(out_fr, label)
                        loss0 += loss
                        val_accuracy += (out_fr.argmax(dim=1) == label.cuda()).float().sum().item()
                        val_num += label.numel() 
                        num += 1
                        progress_bars[1].set_postfix(loss='%.4f' % loss, epoch=epoch, acc=val_accuracy / val_num,
                                                     session=session, num=trial_num)
                        functional.reset_net(net)
                    val_accuracy /= val_num
                    loss0 /= num
                    val_acc.append(val_accuracy)
                    val_loss.append(loss0)
                    logging.info(f"Epoch {epoch}, Validation Accuracy: {val_accuracy:.4f}, Loss: {loss0:.4f}")

                EarlyStop(val_accuracy, net)
                if EarlyStop.early_stop:
                    logging.info("Early stopping at epoch %d", epoch)
                    break

                for progress_bar in progress_bars:
                    progress_bar.refresh()

            for progress_bar in progress_bars:
                progress_bar.close()

            net.load_state_dict(torch.load(model_path))
            test_acc = 0
            test_num = 0
            net.eval()
            with torch.no_grad():
                pbar2 = tqdm(test_data_loader, total=prms['epoch'], desc='Testing')
                for frame, label in pbar2:
                    frame = frame.cuda() 
                    label = label.reshape(-1).cuda()
                    out_fr = net(frame)  # N,num_classes
                    test_acc += (out_fr.argmax(dim=1) == label).float().sum().item()
                    test_num += label.numel() 
                    pbar2.set_postfix(loss='%.4f' % loss, acc=test_acc / test_num, session=session, num=trial_num)
                    functional.reset_net(net)
                test_acc /= test_num
                test_acc = round(test_acc, 4)
                logging.info(f"Test Accuracy for session {session}: {test_acc}")

            session_test_acc.append(100 * test_acc)

        logging.info("Session accuracies: %s", session_test_acc)
        logging.info('Session mean accuracy: %.4f', np.mean(np.array(session_test_acc)))
        trial_acc.append(session_test_acc)

    end_time = time.time()
    trial_acc = np.array(trial_acc)
    logging.info(f'trial_acc: \n {trial_acc}', )
    session_mean = np.mean(trial_acc, axis=0).reshape(-1)
    session_var = np.var(trial_acc, axis=0).reshape(-1)
    logging.info('Session mean accuracies:\n %s', session_mean)
    logging.info('Session standard deviations:\n %s', np.sqrt(session_var))
    trial_mean = np.mean(trial_acc, axis=1).reshape(-1)
    logging.info('Trial mean accuracies:\n %s', trial_mean)
    result_mean = np.mean(trial_mean)
    result_var = np.var(trial_mean)
    logging.info('Final Result:\n %.4f±%.4f', result_mean, np.sqrt(result_var))
    logging.info(f'Total training time:\n {end_time - start_time:.4f} seconds')

    txt_filename = log_dir + filename + f".txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        np.savetxt(f, trial_acc, fmt='%.2f')
        f.write('每个session的均值是：\n')
        np.savetxt(f, session_mean, fmt='%.3f')
        f.write('每个trial的均值是：\n')
        np.savetxt(f, trial_mean, fmt='%.3f')
        f.write(f'实验结果是：\n{result_mean}')
