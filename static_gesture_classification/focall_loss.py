from torch.autograd import Variable
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn


class FocalLoss(nn.Module):
    # Implementation from https://www.kaggle.com/code/zeta1996/pytorch-lightning-arcface-focal-loss
    """
    This criterion is a implemenation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection.

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.

    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
        size_average(bool): By default, the losses are averaged over observations for each minibatch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each minibatch.

    """

    def __init__(
        self,
        class_num: int,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2,
        size_average: bool = True,
    ):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        class_num = inputs.size(1)
        probabilities = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(batch_size, class_num).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data.type(torch.int64), 1.0)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = torch.gather(self.alpha, 0, ids.data.view(-1, 1).type(torch.int64))
        probs = (probabilities * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
