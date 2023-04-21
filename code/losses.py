import torch
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
def OG_CLIP_loss(logits_per_image,  logits_per_text):

    labels = torch.eye(logits_per_image.shape[0], device = logits_per_image.device)
    loss_i = cross_entropy(logits_per_image, labels, reduction='none') #cross_entropy_loss(logits_per_image.softmax(dim = -1), labels)
    loss_e = cross_entropy(logits_per_text, labels, reduction='none') #cross_entropy_loss(logits_per_text.softmax(dim = -1), labels)
    loss = (loss_i + loss_e)/2.0

    return loss.mean()
    
