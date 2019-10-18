from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils import topk_accuracy, ValueMeter, topk_accuracy, get_marginal_indexes, marginalize, softmax,  topk_recall, predictions_to_json
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from network import Network
from torch.optim import lr_scheduler
from torch import nn
import copy

pd.options.display.float_format = '{:05.2f}'.format

parser = ArgumentParser(description="Training program with Multi-Action-Banks")
parser.add_argument('mode', type=str, choices=['train', 'validate', 'test'], default='train',
                    help="Whether to perform training, validation or test.\
                            If test is selected, --json_directory must be used to provide\
                            a directory in which to save the generated jsons.")
parser.add_argument('path_to_data', type=str,
                    help="Path to the data folder, \
                            containing all LMDB datasets")
parser.add_argument('path_to_models', type=str,
                    help="Path to the directory where to save all models")
parser.add_argument('path_to_lmdb_data', type=str, help="Path to lmdb data")
parser.add_argument('--alpha', type=float, default=1,
                    help="Distance between time-steps in seconds")
parser.add_argument('--task', type=str, default='anticipation', choices=[
                    'anticipation', 'early_recognition'], help='Task to tackle: \
                            anticipation or early recognition')
parser.add_argument('--img_tmpl', type=str,
                    default='frame_{:010d}.jpg', help='Template to use to load the representation of a given frame')
parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'flow', 'obj', 'fusion', 'late_fusion'], \
                    help = "Modality. Rgb/flow/obj represent single branches, whereas fusion indicates the whole model with modality attention.")

parser.add_argument('--num_class', type=int, default=2513,
                    help='Number of classes')
parser.add_argument('--batch_size', type=int, default=10, help="Batch Size")
parser.add_argument('--num_workers', type=int, default=4,
                    help="Number of parallel thread to fetch the data")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
parser.add_argument('--schedule_on',     type=int,    default=1,  help='')
parser.add_argument('--schedule_epoch',  type=int,    default=10, help='')


parser.add_argument('--display_every', type=int, default=10,
                    help="Display every n iterations")
parser.add_argument('--epochs', type=int, default=40, help="Training epochs")
parser.add_argument('--ignore_checkpoints', action='store_true',
                    help='If specified, avoid loading existing models (no pre-training)')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume suspended training')

parser.add_argument('--json_directory', type=str, default = None, help = 'Directory in which to save the generated jsons.')
parser.add_argument('--in_dim_past1',        type=int,    default=15,    help='') # 10
parser.add_argument('--in_dim_past2',        type=int,    default=10,    help='') # 
parser.add_argument('--in_dim_past3',        type=int,    default=5,    help='') 
parser.add_argument('--past_sec',    type=int,    default=16,   help='') # 
#    parser.add_argument('--in_dim_past4',        type=int,    default=5,     help='')

parser.add_argument('--current_seconds',     type=int,    default=3,    help='')
parser.add_argument('--current_seconds2',    type=int,    default=2,   help='') # 256, 512
parser.add_argument('--current_seconds3',    type=int,    default=1,   help='') # 256, 512
parser.add_argument('--rel_sec',    type=int,    default=2,   help='') # 256, 512
parser.add_argument('--in_dim_curr',         type=int,    default=3,     help='')
parser.add_argument('--latent_dim',          type=int,    default=512,  help='')
parser.add_argument('--linear_dim',          type=int,    default=512,  help='')
parser.add_argument('--dropout_rate',        type=float,  default=0.3,   help='')
parser.add_argument('--dropout_rate_linear', type=float,  default=0.3,   help='')
parser.add_argument('--video_feat_dim',  type=int,    default=1024,       help='') 
parser.add_argument('--scale_factor',    type=float,  default=-.5,                  help='')
parser.add_argument('--scale',           type=bool,   default=True,                 help='')
parser.add_argument('-l','--fusion_list', action='append', help='<Required> Set flag', required=False)
parser.add_argument('--f_max', action='store_true', help="Fuses 3 streams to single stream")
parser.add_argument('--hsplit', type=int, default=2, help="Splits into stream parts stream")
parser.add_argument('--late_fusion', action='store_true', help=" ")
args = parser.parse_args()

if args.mode == 'test':
    assert args.json_directory is not None

if args.modality == "fusion":
    assert args.fusion_list is not None
    assert len(args.fusion_list) > 1
    print(args.fusion_list)
    print("HSplit array ", args.hsplit)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.task == 'anticipation':

    exp_name = "action_bank_mod_{}_past_{}_rel_{}_cur1_{}_cur2_{}_cur3_{}_inp1_{}_inp2_{}_inp3_{}_inc_{}_sc_{}_bs_{}_drop_{}_max_epoc_{}_feature_dim_{}_latent_{}_linear_{}".format(
                args.modality, args.past_sec, args.rel_sec, args.current_seconds,
                args.current_seconds2, args.current_seconds3,args.in_dim_past1,
                args.in_dim_past2, args.in_dim_past3, args.in_dim_curr,
                args.schedule_epoch, args.batch_size, args.dropout_rate, args.epochs, args.video_feat_dim, args.latent_dim, args.linear_dim)
    if args.modality == "fusion":
        exp_name = exp_name + "_".join(args.fusion_list)

exp_name = exp_name + '_multi_scale_action_bank'

print("Store file name ", exp_name)

print("Printing Arguments ")
print(args)

if args.late_fusion: # Considering args parameters from object model
    args_rgb = copy.deepcopy(args)
    args_rgb.video_feat_dim = 1024
    args_rgb.current_seconds = 8
    args_rgb.current_seconds2 = 6
    args_rgb.current_seconds3 = 3
    args_rgb.latent_dim = 512
    args_rgb.linear_dim = 512
    args_rgb.dropout_rate = 0.3
    args_rgb.dropout_rate_linear = 0.3
    args_rgb.epochs = 40
    args_rgb.batch_size = 10
    if args.modality == "fusion":
        exp_name = exp_name + "_".join(args.fusion_list)
    exp_rgb_name = "action_bank_mod_{}_past_{}_rel_{}_cur1_{}_cur2_{}_cur3_{}_inp1_{}_inp2_{}_inp3_{}_inc_{}_sc_{}_bs_{}_drop_{}_max_epoc_{}_feature_dim_{}_latent_{}_linear_{}".format(
                args_rgb.modality, args_rgb.past_sec, args_rgb.rel_sec, args_rgb.current_seconds,
                args_rgb.current_seconds2, args_rgb.current_seconds3,args_rgb.in_dim_past1,
                args_rgb.in_dim_past2, args_rgb.in_dim_past3, args_rgb.in_dim_curr,
                args_rgb.schedule_epoch, args_rgb.batch_size, args.dropout_rate, args.epochs, args_rgb.video_feat_dim, args_rgb.latent_dim, args_rgb.linear_dim)
    exp_rgb_name += '_multi_scale_action_bank'

    args_flow     = copy.deepcopy(args_rgb)
    exp_flow_name = "action_bank_mod_{}_past_{}_rel_{}_cur1_{}_cur2_{}_cur3_{}_inp1_{}_inp2_{}_inp3_{}_inc_{}_sc_{}_bs_{}_drop_{}_max_epoc_{}_feature_dim_{}_latent_{}_linear_{}".format(
                     args_flow.modality, args_flow.past_sec, args_flow.rel_sec, args_flow.current_seconds,
                     args_flow.current_seconds2, args_flow.current_seconds3, args_flow.in_dim_past1,
                     args_flow.in_dim_past2, args_flow.in_dim_past3, args_flow.in_dim_curr,
                     args_flow.schedule_epoch, args_flow.batch_size, args_flow.dropout_rate, args_flow.epochs, args_flow.video_feat_dim, args_flow.latent_dim, args_flow.linear_dim)
    exp_flow_name += '_multi_scale_action_bank'


def get_loader(mode, override_modality = None):
    if override_modality:
        path_to_lmdb = join(args.path_to_lmdb_data, override_modality)
    else:
        path_to_lmdb = join(args.path_to_lmdb_data, args.modality) if args.modality != 'fusion' else [join(args.path_to_lmdb_data, m) for m in args.fusion_list]

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, "{}.csv".format(mode)),
        'time_step': args.alpha,
        'img_tmpl': args.img_tmpl,
        'past_features': args.task == 'anticipation',
        'sequence_length': 1, #args.S_enc + args.S_ant,
        'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
        'challenge': 'test' in mode,
        'args': args
    }

    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training')


def get_model():
    if not args.late_fusion:
        return Network(args)
    elif args.late_fusion:
        obj_model = Network(args)
        rgb_model = Network(args_rgb)
        flow_model = Network(args_flow)
        checkpoint_rgb = torch.load(join(args.path_to_models,\
                  exp_rgb_name.replace('late_fusion','rgb') +'_best.pth.tar'))['state_dict']
        checkpoint_flow = torch.load(join(args.path_to_models,\
                    exp_flow_name.replace('late_fusion','flow') +'_best.pth.tar'))['state_dict']
        checkpoint_obj = torch.load(join(args.path_to_models,\
                 exp_name.replace('late_fusion','obj') +'_best.pth.tar'))['state_dict']
        rgb_model.load_state_dict(checkpoint_rgb)
        flow_model.load_state_dict(checkpoint_flow)
        obj_model.load_state_dict(checkpoint_obj)
        return [rgb_model, flow_model, obj_model]
    

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


def log(mode, epoch, loss_meter, accuracy_meter, accuracy_future, accuracy_future1, accuracy_future2, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")
    print(
            "[{}] Epoch: {:.2f}. ".format(mode, epoch),
            "Loss: {:.2f}. ".format(loss_meter.value()),
            "Accuracy: {:.2f}% ".format(accuracy_meter.value()),
            "Accuracy Future: {:.2f}% ".format(accuracy_future.value()),
            "Accuracy Future1: {:.2f}% ".format(accuracy_future1.value()),
            "Accuracy Future2: {:.2f}% ".format(accuracy_future2.value()), end="")

    if best_perf:
        print("[best: {:.2f}]%".format(best_perf), end="")

    print('\033[0m')


def get_scores_late_fusion(models, loaders):
    verb_scores = []
    noun_scores = []
    action_scores = []
    for model, loader in zip(models, loaders):
        outs = get_scores(model, loader)
        verb_scores.append(outs[0])
        noun_scores.append(outs[1])
        action_scores.append(outs[2])

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
            x = batch['past_features' if args.task == 'anticipation' else 'action_features']
            x_recent = batch['recent_features' if args.task == 'anticipation' else 'action_features']
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

            predict_future, predict_future2, predict_future3 = model(x, x_recent)
            preds = predict_future.detach().cpu().numpy() + predict_future2.detach().cpu().numpy() + predict_future3.detach().cpu().numpy()

            predictions.append(preds)
            labels.append(y)

    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')

    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi) # .reshape( action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni) # .reshape( action_scores.shape[0], action_scores.shape[1], -1)

    
    if labels.max() > 0:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2]
    else:
        return verb_scores, noun_scores, action_scores, ids


def trainval(model, loaders, optimizer, epochs, start_epoch, start_best_perf, schedule_on):
    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch
    loss_func_future = nn.CrossEntropyLoss()
    loss_func_future2 = nn.CrossEntropyLoss()
    loss_func_future3 = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, epochs):
        if schedule_on is not None:
            schedule_on.step()
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_future_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_future1_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_future2_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    x = batch['past_features' if args.task ==
                              'anticipation' else 'action_features']
                    x_recent = batch['recent_features' if args.task ==
                              'anticipation' else 'action_features']
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

                    predict_future, predict_future2, predict_future3 = model(x, x_recent)

                    loss = loss_func_future(predict_future, y) + loss_func_future2(predict_future2, y) + loss_func_future3(predict_future3, y)

                    # use top-5 for anticipation and top-1 for early recognition
                    k = 5 if args.task == 'anticipation' else 1

                    acc_future = topk_accuracy(predict_future.detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0]*100
                    accuracy_future_meter[mode].add(acc_future, bs)
                    acc_future1 = topk_accuracy(predict_future2.detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0]*100
                    accuracy_future1_meter[mode].add(acc_future1, bs)
                    acc_future2 = topk_accuracy(predict_future3.detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0]*100
                    accuracy_future2_meter[mode].add(acc_future2, bs)

                    preds = predict_future.detach() + predict_future2.detach() + predict_future3.detach()
                    acc = topk_accuracy(preds.detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0] * 100

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
                        log(mode, e, loss_meter[mode], accuracy_meter[mode], accuracy_future_meter[mode], accuracy_future1_meter[mode], accuracy_future2_meter[mode])

                # log at the end of each epoch
                log(mode, epoch + 1, loss_meter[mode], accuracy_meter[mode], accuracy_future_meter[mode], accuracy_future1_meter[mode], accuracy_future2_meter[mode],
                    max(accuracy_meter[mode].value(), best_perf) if mode == 'validation' else None, green=True)

        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(model, epoch + 1, accuracy_meter['validation'].value(), best_perf, is_best=is_best)

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
    if type(model) == list:
        model = [m.to(device) for m in model]
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
        if args.late_fusion:
            loaders = [get_loader('validation', 'rgb'), get_loader('validation', 'flow'), get_loader('validation', 'obj')]
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores_late_fusion(model, loaders)
        else:
            epoch, perf, _ = load_checkpoint(model, best=True)
            print("Loaded checkpoint for model {}. Epoch: {}. Perf: {:0.2f}.".format(type(model), epoch, perf))
            
            loader = get_loader('validation')

            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores(model, loader)

        verb_accuracies = topk_accuracy(verb_scores, verb_labels, (5,))[0]
        noun_accuracies = topk_accuracy(noun_scores, noun_labels, (5,))[0]
        action_accuracies = topk_accuracy(action_scores, action_labels, (5,))[0]

        many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot()

        verb_recalls = topk_recall(verb_scores, verb_labels, k=5, classes=many_shot_verbs)
        noun_recalls = topk_recall(noun_scores, noun_labels, k=5, classes=many_shot_nouns)
        action_recalls = topk_recall(action_scores, action_labels, k=5, classes=many_shot_actions)
        print("Verb Accuracy ", verb_accuracies * 100)
        print("Noun Accuracy ", noun_accuracies * 100)
        print("Action Accuracy ", action_accuracies * 100)
        print("Verb Recall", verb_recalls * 100)
        print("Noun Recall", noun_recalls * 100)
        print("Action Recall", action_recalls * 100)

    elif args.mode == 'test':
        for m in ['seen','unseen']:
            if args.late_fusion:
                loaders = [get_loader("test_{}".format(m), 'rgb'), get_loader("test_{}".format(m), 'flow'), get_loader("test_{}".format(m), 'obj')]
                discarded_ids = loaders[0].dataset.discarded_ids
                verb_scores, noun_scores, action_scores, ids = get_scores_late_fusion(model, loaders)
            else:
                loader = get_loader("test_{}".format(m))
                epoch, perf, _ = load_checkpoint(model, best=True)

                discarded_ids = loader.dataset.discarded_ids

                print("Loaded checkpoint for model {}. Epoch: {}. Perf: {:.2f}.".format(type(model), epoch, perf))

                verb_scores, noun_scores, action_scores, ids = get_scores(model, loader)

            idx = -4 if args.task == 'anticipation' else -1
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
