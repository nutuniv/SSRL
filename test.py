import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from misc import all_gather,is_main_process,get_world_size,synchronize


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def test(dataloader, model, args,  device,logger):
    with torch.no_grad():
        model.eval()
        pred = {}
        for i, (input,index) in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred.update({index.cpu().detach().numpy()[0]:list(sig.cpu().detach().numpy())})
            torch.cuda.empty_cache()
        
        synchronize()
        predictions = _accumulate_predictions_from_multiple_gpus(pred)
        if not is_main_process():
            return
        pred = []
        for i in range(len(predictions)):
            pred = pred + predictions[i]

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        logger.info('[test]: auc\t{:.4f}'.format( rec_auc))
        precision, recall, th = precision_recall_curve(list(gt), pred)
        return rec_auc

