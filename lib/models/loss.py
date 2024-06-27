import torch
import torch.nn.functional as F

def attention_loss(batch_attentions, targets):
    # (b, seq, xxx)
    loss_attn = 0
    num = 0
    for i in range(len(targets)):  # batch
        seq_attentions = batch_attentions[i]
        num += len(seq_attentions)
        target = targets[i]
        seq, num_clips = seq_attentions.shape
        gt_attention = torch.zeros(seq_attentions.shape).to(seq_attentions.device)
        for j in range(seq):  # seq
            s = target[j][0]
            e = target[j][1]
            for k in range(num_clips):
                t = 1.0 * k / num_clips
                if s <= t <= e:
                    gt_attention[j, k] = 1
        loss_attn = loss_attn + (-torch.log2(torch.sum(seq_attentions * gt_attention, 1) + 1e-10).sum())

    return loss_attn / num


def bboxes_F1_loss(boxes, targets):
    loss_bbox = 0
    m = 0
    for i in range(len(boxes)):
        m = m + len(boxes[i])
        loss = F.l1_loss(boxes[i], torch.tensor(targets[i]).float().to(boxes[i].device), reduction='sum')
        loss_bbox += loss
    return loss_bbox / m


def giou_loss(boxes, targets):
    loss_giou = 0
    m = 0
    for i in range(len(boxes)):
        m = m + len(boxes[i])
        for j in range(len(boxes[i])):
            inter_left = max(boxes[i][j][0], targets[i][j][0])
            inter_right = min(boxes[i][j][1], targets[i][j][1])
            inter = max(0.0, inter_right - inter_left)
            union_left = min(boxes[i][j][0], targets[i][j][0])
            union_right = max(boxes[i][j][1], targets[i][j][1])
            union = max(0.0, union_right - union_left)
            iou = 1.0 * inter / union
            loss_giou = loss_giou + 1 - iou
    return loss_giou / m


def final_loss(srcs, targets, cfg, attention):
    loss_attn = attention_loss(attention, targets)
    loss_bbox = bboxes_F1_loss(srcs, targets)
    loss_giou = giou_loss(srcs, targets)
    loss = loss_attn + cfg.LAMBDA * loss_bbox + cfg.BETA * loss_giou
    return loss, loss_attn, loss_bbox, loss_giou


