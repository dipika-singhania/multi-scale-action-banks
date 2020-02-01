from argparse import ArgumentParser
from dataset import SequenceDataset, my_collate
from os.path import join
import torch
from torch.utils.data import DataLoader
from utils import ValueMeter, topk_accuracy, get_marginal_indexes, marginalize, softmax,  topk_recall, predictions_to_json
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from network import Network
from torch.optim import lr_scheduler
from torch import nn
import copy
COMP_PATH =  '/mnt/auersberg/Anticipation/'
import os
#COMP_PATH = '/home/sener/share/Anticipation/'

pd.options.display.float_format = '{:05.2f}'.format

parser = ArgumentParser(description="Training program with Multi-Action-Banks")
parser.add_argument('mode', type=str,   default='validate', choices=['train', 'validate', 'train_val', 'test'], help="Whether to perform training, validation or test. If test is selected, --json_directory must be used to provide a directory in which to save the generated jsons.")
parser.add_argument('path_to_data',     type=str,   help="Path to the data folder,  containing all LMDB datasets")
parser.add_argument('path_to_models',   type=str,   help="Path to the directory where to save all models")
#parser.add_argument('--mode',             type=str,   default='validate', choices=['train', 'validate', 'train_val', 'test'], help="Whether to perform training, validation or test. If test is selected, --json_directory must be used to provide a directory in which to save the generated jsons.")
#parser.add_argument('--path_to_data',     type=str,   default=COMP_PATH + 'EPIC_CODES/DATA_EPIC/',  help="Path to the data folder,  containing all LMDB datasets")
#parser.add_argument('--path_to_models',   type=str,   default=COMP_PATH + 'EPIC_CODES/models/', help="Path to the directory where to save all models")

parser.add_argument('--json_directory',  type=str,   default = None, help = 'Directory in which to save the generated jsons.')
parser.add_argument('--task',            type=str,   default='anticipation', choices=['anticipation', 'recognition'], help='Task to tackle: anticipation or early recognition')
parser.add_argument('-a', '--alpha',     action='append', help="Distance between time-steps in seconds")
parser.add_argument('--img_tmpl',        type=str,   default='frame_{:010d}.jpg', help='Template to use to load the representation of a given frame')
parser.add_argument('-l','--fusion_list',action='append', help='<Required> Set flag', required=False)
parser.add_argument('--resume',          action='store_true', help='Whether to resume suspended training')

parser.add_argument('--modality',        type=str,   default='late_fusion', choices=['rgb', 'flow', 'obj', 'fusion', 'late_fusion', 'audio', 'rgb_resnet', 'rgb_RU', 'rgb_BN_TBN'],  help = "Modality. Rgb/flow/obj represent single branches, whereas fusion indicates the whole model with modality attention.")
parser.add_argument('--best_model',      type=str,  default='best', help='') # 'best' 'last'
parser.add_argument('--weight_rgb',      type=float, default=5, help='')
parser.add_argument('--weight_rgb_RU',      type=float, default=5, help='')
parser.add_argument('--weight_rgb_BN_TBN',      type=float, default=5, help='')
parser.add_argument('--weight_flow',     type=float, default=5, help='')
parser.add_argument('--weight_obj',      type=float, default=5, help='')
parser.add_argument('--weight_audio',    type=float, default=0, help='')

parser.add_argument('--num_class',       type=int,   default=2513, help='Number of classes')
parser.add_argument('--num_workers',     type=int,   default=4,    help="Number of parallel thread to fetch the data")
parser.add_argument('--display_every',   type=int,   default=10,   help="Display every n iterations")
parser.add_argument('--rel_sec',         type=int,   default=1,    help='') #

parser.add_argument('--schedule_on',     type=int,   default=1,    help='')
parser.add_argument('--schedule_epoch',  type=int,   default=10,   help='')
parser.add_argument('--scale_factor',    type=float, default=-.5,  help='')
parser.add_argument('--scale',           type=bool,  default=True, help='')
parser.add_argument('--hsplit',          type=int,   default=2,    help="Splits into stream parts stream")
parser.add_argument('--f_max',           action='store_true', help="Fuses 3 streams to single stream")

parser.add_argument('--video_feat_dim',  type=int,   default=352, help='') # 352 1024
parser.add_argument('--lr',              type=float, default=1e-4, help="Learning rate")
parser.add_argument('--batch_size',      type=int,   default=10,   help="Batch Size")
parser.add_argument('--epochs',          type=int,   default=45,   help="Training epochs")

parser.add_argument('--past_sec',        type=float, default=10,    help='') #
parser.add_argument('-p','--dim_past_list', action='append', type=int, help='Past seconds to be taken into account', required=False)

parser.add_argument('--dim_curr',        type=int,   default=2,    help='')
parser.add_argument('-c','--curr_sec_list', action='append', type=float, help='current seconds to be taken into account', required=False)

parser.add_argument('--latent_dim',      type=int,   default=512,  help='')
parser.add_argument('--linear_dim',      type=int,   default=512,  help='')
parser.add_argument('--dropout_rate',    type=float, default=0.3,  help='')
parser.add_argument('--dropout_linear',  type=float, default=0.3,  help='')
parser.add_argument('--add_verb_loss',   action='store_true', help='Whether to train with verb loss or not')
parser.add_argument('--verb_class', type=int, default=125, help='')
parser.add_argument('--add_noun_loss',   action='store_true', help='Whether to train with verb loss or not')
parser.add_argument('--noun_class', type=int, default=352, help='')
parser.add_argument('-s', '--samples_envs', action='append', type=str, help='path_to_data=/mnt/data/epic_kitchen/data/,\
                     -s=sample1 new_path=/mnt/data/epic_kitchen/sample1/', required=False)
parser.add_argument('--take_future', action='store_true', help="Whether to consider -5+5 etc combination")
parser.add_argument('--val_sec', type=float, help="Alpha for validation sec", required=False)
args = parser.parse_args()

if len(args.curr_sec_list) == 0:
    args.curr_sec_list = [0, 0.5, 1, 1.5]

if len(args.dim_past_list) == 0:
    args.dim_past_list = [5, 3, 2]

string_of_env = ''
list_of_envs = []
if args.samples_envs is not None and len(args.samples_envs) != 0:
    dirname             = os.path.dirname(args.path_to_data)
    base_directory_name = os.path.basename(dirname)
    list_of_envs        = [args.path_to_data]
    string_of_env       = base_directory_name
    for env_name in args.samples_envs:
        k = args.path_to_data.rfind(base_directory_name)
        new_directory   = args.path_to_data[:k] + env_name + "/"
        assert os.path.exists(new_directory), 'Please specify correct directory, ' + new_directory + ' does not exost'
        string_of_env  += "_" + env_name
        list_of_envs.append(new_directory)
    print("Using list of environments = ", list_of_envs)

print(args)

if args.mode == 'test':
    assert args.json_directory is not None

if args.modality == "fusion":
    assert args.fusion_list is not None
    assert len(args.fusion_list) > 1
    print(args.fusion_list)
    print("HSplit array ", args.hsplit)

if args.modality == "late_fusion":
    if args.fusion_list is None:
        args.fusion_list = ['rgb', 'flow', 'obj']
    print("Taking modalities = ", args.fusion_list)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_model_name(argument):
    exp_name = "ab_task_{}_mod_{}_past_{}".format(argument.task, argument.modality, argument.past_sec)
    exp_name = exp_name + "_" + "_".join(['dp{}_{}'.format(i, num) for i, num in enumerate(argument.dim_past_list)])
    exp_name = exp_name + "_dc_{}".format(argument.dim_curr)
    exp_name = exp_name + "_" + "_".join(['cur{}_{}'.format(i, num) for i, num in enumerate(argument.curr_sec_list)])
    exp_name = exp_name + "_sch_{}_bs_{}_ep_{}_drn_{}_drl_{}_lr_{}_dimLa_{}_dimLi_{}".format(
                argument.schedule_epoch, argument.batch_size, argument.epochs, argument.dropout_rate,
                argument.dropout_linear, argument.lr, argument.latent_dim, argument.linear_dim)
    if argument.modality == "fusion":
        exp_name = exp_name + "_".join(argument.fusion_list)
    if argument.add_verb_loss:
        exp_name = exp_name + '_vb_ls'
    if argument.add_noun_loss:
        exp_name = exp_name + '_nn_ls'
    if len(string_of_env) != 0:
        exp_name = exp_name + '_' + string_of_env
    if argument.take_future:
        exp_name = exp_name + "_tf"
    if argument.alpha and args.task == "anticipation":
        if isinstance(argument.alpha, list):
            exp_name = exp_name + "_".join(map(str, argument.alpha))
        else:
            exp_name = exp_name + str(argument.alpha)
    if argument.path_to_data:        
        dirname = os.path.dirname(args.path_to_data)
        base_directory_name = os.path.basename(dirname)
        exp_name = exp_name + base_directory_name

    exp_name = exp_name + '_ms_bank'
    return exp_name


exp_name = make_model_name(args)
print("Store file name ", exp_name)

if args.modality == 'late_fusion': # Considering args parameters from object model
    assert (args.mode != 'train' and args.mode != 'train_val')
    args_rgb = copy.deepcopy(args)
    args_rgb.video_feat_dim = 1000
    args_rgb.dim_curr = 2
    exp_rgb_name      = make_model_name(args_rgb)

    args_rgb_RU = copy.deepcopy(args)
    args_rgb_RU.video_feat_dim = 1024
    args_rgb_RU.dim_curr = 2
    exp_rgb_name_RU = make_model_name(args_rgb_RU)

    args_rgb_BN_TBN = copy.deepcopy(args)
    args_rgb_BN_TBN.video_feat_dim = 1024
    args_rgb_BN_TBN.dim_curr = 2
    exp_rgb_name_BN_TBN = make_model_name(args_rgb_BN_TBN)

    args_flow          = copy.deepcopy(args_rgb)
    args_flow.video_feat_dim = 1024
    args_flow.dim_curr = 2
    exp_flow_name      = make_model_name(args_flow)

    args_audio          = copy.deepcopy(args_rgb)
    args_audio.video_feat_dim = 1024
    args_audio.dim_past_list = [5, 3, 2]
    args_audio.dim_curr = 2
    args_audio.curr_sec_list = [0, 1, 2]
    args_audio.past_sec = 10
    exp_audio_name      = make_model_name(args_audio)

    args_dict = {'rgb': args_rgb, 'flow': args_flow, 'audio': args_audio, 'obj': args, 'rgb_RU': args_rgb_RU, 'rgb_BN_TBN': args_rgb_RU}
    args_name_dict = {'rgb': exp_rgb_name, 'flow': exp_flow_name, 'audio': exp_audio_name, 'obj': exp_name, \
                      'rgb_RU': exp_rgb_name_RU, 'rgb_BN_TBN': exp_rgb_name_BN_TBN}
    weights_dict = {'rgb': args.weight_rgb, 'flow': args.weight_flow, 'audio': args.weight_audio, 'obj':args.weight_obj, 'rgb_RU': args.weight_rgb_RU,
                    'rgb_BN_TBN': args.weight_rgb_BN_TBN}

def get_loader(mode, override_modality = None):
    global list_of_envs
    multi_samples = 0
    if override_modality:
        path_to_lmdb = join(args.path_to_data, override_modality)
    else:
        if len(list_of_envs) > 0 and mode == 'training':
            path_to_lmdb = []
            for envs in list_of_envs:
                ele_of_list = join(envs, args.modality) if args.modality != 'fusion' else [join(args.path_to_data, m) for m in args.fusion_list]
                path_to_lmdb.append(ele_of_list)
            multi_samples = len(list_of_envs)
        else:
            path_to_lmdb = join(args.path_to_data, args.modality) if args.modality != 'fusion' else [join(args.path_to_data, m) for m in args.fusion_list]
            multi_samples = 0

    if mode != 'training' and args.val_sec is not None:
        alpha = args.val_sec
    else:
        alpha = args.alpha

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, "{}.csv".format(mode)),
        'time_step': alpha,
        'img_tmpl': args.img_tmpl,
        'task': args.task,
        'sequence_length': 1,
        'label_type': ['verb', 'noun', 'action'],
        'multi_samples': multi_samples,
        'challenge': 'test' in mode,
        'args': args
    }

    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training', collate_fn=my_collate)


def get_model():
    if not args.modality == 'late_fusion' :
        return Network(args)
    elif args.modality == 'late_fusion':
        model_dict = {}
        for mode_f in args.fusion_list:
            model_dict[mode_f] = Network(args_dict[mode_f])
            end_path = '.pth.tar'
            if args.best_model == 'best':
                print('args.best_model == True')
                end_path = '_best.pth.tar'
            checkpoint = torch.load(join(args.path_to_models, args_name_dict[mode_f].replace('late_fusion', mode_f) + end_path))['state_dict']
            model_dict[mode_f].load_state_dict(checkpoint)
        return model_dict



def load_checkpoint(model, best=False):
    if best:
        chk = torch.load(join(args.path_to_models, exp_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(args.path_to_models, exp_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(model, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.path_to_models, exp_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.path_to_models, exp_name + '_best.pth.tar'))


def log(mode, epoch, loss_meter, accuracy_meter, accuracy_future_list, action_loss, verb_loss, noun_loss, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")
    str_accuracy = []
    for i, accuracy_future in enumerate(accuracy_future_list):
        str_accuracy.append("Accuracy Future{}: {:.2f}% ".format(i, accuracy_future.value()))
    print(
            "[{}] Epoch: {:.2f}. ".format(mode, epoch),
            "Total Loss: {:.2f}. ".format(loss_meter.value()),
            "Action Loss: {:.2f}. ".format(action_loss.value()),
            "Verb Loss: {:.2f}. ".format(verb_loss.value()),
            "Noun Loss: {:.2f}. ".format(noun_loss.value()),
            "Accuracy: {:.2f}% ".format(accuracy_meter.value()),
            ",".join(str_accuracy), end="")

    if best_perf:
        print("[best: {:.2f}]%".format(best_perf), end="")

    print('\033[0m')


def get_scores_late_fusion(models, loaders):
    verb_scores = []
    noun_scores = []
    action_scores = []
    for mode_f in args.fusion_list:
        outs = get_scores(models[mode_f], loaders[mode_f])
        verb_scores.append(outs[0] * weights_dict[mode_f])
        noun_scores.append(outs[1] * weights_dict[mode_f])
        action_scores.append(outs[2] * weights_dict[mode_f])

    verb_scores = sum(verb_scores)
    noun_scores = sum(noun_scores)
    action_scores = sum(action_scores)

    return [verb_scores, noun_scores, action_scores] + list(outs[3:])


def get_scores(model, loader):
    model.eval()
    predictions = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['past_features']
            x_recent = batch['recent_features']
            if type(x) == list:
                x = [xx.to(device) for xx in x]
            else:
                x = x.to(device)

            if type(x_recent) == list:
                x_recent = [xx.to(device) for xx in x_recent]
            else:
                x_recent = x_recent.to(device)

            y = batch['label'].numpy()

            ids.append(batch['id'].numpy())

            predict_future_list, predict_verb_list, predict_noun_list = model(x, x_recent)

            preds = 0
            for pred_f in predict_future_list:
                preds +=  pred_f.detach().cpu().numpy()

            predictions.append(preds)
            labels.append(y)

    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')

    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi) 
    noun_scores = marginalize(action_probs, ni) 


    if labels.max() > 0:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2]
    else:
        return verb_scores, noun_scores, action_scores, ids


def trainval(model, loaders, optimizer, epochs, start_epoch, start_best_perf, schedule_on):
    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch
    loss_func_future_list = [nn.CrossEntropyLoss() for _ in range(len(args.curr_sec_list))]
    if args.add_verb_loss:
        loss_func_verb_list = [nn.CrossEntropyLoss() for _ in range(len(args.curr_sec_list))]
    if args.add_noun_loss:
        loss_func_noun_list = [nn.CrossEntropyLoss() for _ in range(len(args.curr_sec_list))]

    for epoch in range(start_epoch, epochs):
        if schedule_on is not None:
            schedule_on.step()
        # define training and validation meters
        loss_meter                 = {'training': ValueMeter(), 'validation': ValueMeter()}
        action_loss_meter          = {'training': ValueMeter(), 'validation': ValueMeter()}
        verb_loss_meter            = {'training': ValueMeter(), 'validation': ValueMeter()}
        noun_loss_meter            = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_meter             = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_future_meter_list = {'training': [ValueMeter() for _ in range(len(args.curr_sec_list))], 
                                      'validation': [ValueMeter() for _ in range(len(args.curr_sec_list))]}

        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    x = batch['past_features']
                    x_recent = batch['recent_features'] 
                    if type(x) == list:
                        x = [xx.to(device) for xx in x]
                    else:
                        x = x.to(device)

                    if type(x_recent) == list:
                        x_recent = [xx.to(device) for xx in x_recent]
                    else:
                        x_recent = x_recent.to(device)


                    y = batch['label'].to(device)

                    bs = y.shape[0]  # batch size

                    predict_future_list, predict_verb_list, predict_noun_list = model(x, x_recent)


                    loss = 0
                    for j in range(len(predict_future_list)):
                        loss += loss_func_future_list[j](predict_future_list[j], y[:, 2])
                    action_loss_meter[mode].add(loss.item(), bs)

                    if args.add_verb_loss:
                        verb_loss = 0
                        for j in range(len(predict_verb_list)):
                            verb_loss += loss_func_verb_list[j](predict_verb_list[j], y[:, 0])
                        verb_loss_meter[mode].add(verb_loss.item(), bs)
                        loss      = loss + verb_loss
                    else:
                        verb_loss_meter[mode].add(-1, bs)
                        

                    if args.add_noun_loss:
                        noun_loss = 0
                        for j in range(len(predict_noun_list)):
                            noun_loss += loss_func_noun_list[j](predict_noun_list[j], y[:, 1])
                        noun_loss_meter[mode].add(noun_loss.item(), bs)
                        loss      = loss + noun_loss
                    else:
                        noun_loss_meter[mode].add(-1, bs)


                    # use top-5
                    k = 5

                    preds = 0
                    for j, pred_f in enumerate(predict_future_list):
                        acc_future  = topk_accuracy(pred_f.detach().cpu().numpy(), y[:, 2].detach().cpu().numpy(), (k,))[0] * 100
                        accuracy_future_meter_list[mode][j].add(acc_future, bs)
                        preds += pred_f.detach()

                    acc = topk_accuracy(preds.detach().cpu().numpy(), y[:, 2].detach().cpu().numpy(), (k,))[0] * 100

                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    accuracy_meter[mode].add(acc, bs)

                    # if in training mode
                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i / len(loaders[mode])

                    # log training during loop
                    # avoid logging the very first batch. It can be biased.
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode], accuracy_meter[mode], accuracy_future_meter_list[mode], 
                            action_loss_meter[mode], verb_loss_meter[mode], noun_loss_meter[mode])

                # log at the end of each epoch
                log(mode, epoch + 1, loss_meter[mode], accuracy_meter[mode], accuracy_future_meter_list[mode], action_loss_meter[mode], 
                    verb_loss_meter[mode], noun_loss_meter[mode], max(accuracy_meter[mode].value(), best_perf) if mode == 'validation' else None, green=True)


        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False
        with open( args.path_to_models + '/'  +    exp_name + '.txt', 'a') as f:
            f.write("%d - %0.2f\n" %  (epoch + 1, accuracy_meter['validation'].value()) )

        # save checkpoint at the end of each train/val epoch
        save_model(model, epoch + 1, accuracy_meter['validation'].value(), best_perf, is_best=is_best)

    with open( args.path_to_models + '/'  +    exp_name + '.txt', 'a') as f:
        f.write("%d - %0.2f\n" %  (epoch + 1, best_perf))



def get_many_shot():
    """Get many shot verbs, nouns and actions for class-aware metrics (Mean Top-5 Recall)"""
    # read the list of many shot verbs
    many_shot_verbs = pd.read_csv(join(args.path_to_data, 'EPIC_many_shot_verbs.csv'))['verb_class'].values
    # read the list of many shot nouns
    many_shot_nouns = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_nouns.csv'))['noun_class'].values

    # read the list of actions
    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    # create the list of many shot actions
    # an action is "many shot" if at least one
    # between the related verb and noun are many shot
    many_shot_actions = []
    for a, (v, n) in a_to_vn.items():
        if v in many_shot_verbs or n in many_shot_nouns:
            many_shot_actions.append(a)

    return many_shot_verbs, many_shot_nouns, many_shot_actions


def main():
    model = get_model()
    if type(model) == dict:
        for m in model.keys():
            model[m].to(device)
    else:
        model.to(device)

    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}

        if args.resume:
            start_epoch, _, start_best_perf = load_checkpoint(model)
        else:
            start_epoch = 0
            start_best_perf = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.99, 0.999))
        schedule_on = None
        if args.schedule_on:
            schedule_on = lr_scheduler.StepLR(optimizer, args.schedule_epoch, gamma=0.1, last_epoch=-1)

        trainval(model, loaders, optimizer, args.epochs, start_epoch, start_best_perf, schedule_on)

    elif args.mode == 'validate':
        if args.modality == 'late_fusion':
            loaders = {}
            for fuse_mode in args.fusion_list:
                loaders[fuse_mode] = get_loader('validation', fuse_mode)
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores_late_fusion(model, loaders)
        else:
            epoch, perf, _ = load_checkpoint(model, best=True)
            print("Loaded checkpoint for model {}. Epoch: {}. Perf: {:0.2f}.".format(type(model), epoch, perf))
            loader = get_loader('validation')
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores(model, loader)

        verb_accuracies   = topk_accuracy(verb_scores, verb_labels, (5,))[0]
        noun_accuracies   = topk_accuracy(noun_scores, noun_labels, (5,))[0]
        action_accuracies = topk_accuracy(action_scores, action_labels, (5,))[0]

        many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot()

        verb_recalls   = topk_recall(verb_scores, verb_labels, k=5, classes=many_shot_verbs)
        noun_recalls   = topk_recall(noun_scores, noun_labels, k=5, classes=many_shot_nouns)
        action_recalls = topk_recall(action_scores, action_labels, k=5, classes=many_shot_actions)
        print("Verb Accuracy ", verb_accuracies * 100)
        print("Noun Accuracy ", noun_accuracies * 100)
        print("Action Accuracy ", action_accuracies * 100)
        print("Verb Recall", verb_recalls * 100)
        print("Noun Recall", noun_recalls * 100)
        print("Action Recall", action_recalls * 100)

    elif args.mode == 'test':
        for m in ['seen','unseen']:
            if args.modality == 'late_fusion':
                loaders = {}
                for fuse_mode in args.fusion_list:
                    loaders[fuse_mode] = get_loader("test_{}".format(m), fuse_mode)
                discarded_ids = loaders[args.fusion_list[0]].dataset.discarded_ids
                verb_scores, noun_scores, action_scores, ids = get_scores_late_fusion(model, loaders)
            else:
                loader = get_loader("test_{}".format(m))
                epoch, perf, _ = load_checkpoint(model, best=True)
                discarded_ids = loader.dataset.discarded_ids
                print("Loaded checkpoint for model {}. Epoch: {}. Perf: {:.2f}.".format(type(model), epoch, perf))
                verb_scores, noun_scores, action_scores, ids = get_scores(model, loader)

            ids = list(ids) + list(discarded_ids)
            verb_scores = np.concatenate((verb_scores, np.zeros((len(discarded_ids), *verb_scores.shape[1:]))))
            noun_scores = np.concatenate((noun_scores, np.zeros((len(discarded_ids), *noun_scores.shape[1:]))))
            action_scores = np.concatenate((action_scores, np.zeros((len(discarded_ids), *action_scores.shape[1:]))))

            actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
            # map actions to (verb, noun) pairs
            a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
                       for a in actions.iterrows()}

            preds = predictions_to_json(verb_scores, noun_scores, action_scores, ids, a_to_vn)

            with open(join(args.json_directory,exp_name + "_{}.json".format(m)), 'w') as f:
                f.write(json.dumps(preds, indent=4, separators=(',',': ')))

if __name__ == '__main__':
    main()
