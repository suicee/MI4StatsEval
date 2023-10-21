from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EPS = 1e-6
#MI estimators from On [Variational Bounds of Mutual Information] and [UNDERSTANDING THE LIMITATIONS OF VARIATIONALMUTUAL INFORMATION ESTIMATORS]

#Resources:
#https://github.com/google-research/google-research/tree/master/vbmi
#https://github.com/arashabzd/milib/blob/master/milib/mi/estim.py
#https://github.com/ermongroup/smile-mi-estimator/blob/master/estimators.py


def batch_to_score(batch_x, batch_y, MI_net):
    '''
    transform the input variable x batch [bs_size,feature_size] and variable y batch [bs_size,feature_size] into score Matrix [bs_size,bs_size]
    '''
    batch_size = batch_x.size()[0]

    # feature_size=batch_x.size()[1]

    #add dim in batches so that we can combine input batches into a [bs_size,bs_size,feature_size] tensor
    batch_x_ = batch_x.view(batch_size, 1, -1).expand(-1, batch_size, -1)
    batch_y_ = batch_y.view(1, batch_size, -1).expand(batch_size, -1, -1)

    score = MI_net(batch_x_, batch_y_)

    # pos_neg_pair=torch.cat((batch_pos_,batch_neg_),dim=2).view(batch_size,bs_size,-1)
    assert score.size()[-1] == 1, f"wrong score dim:{score.size()}"
    score_matrix = score.squeeze(-1)

    return score_matrix

def batch_to_score_with_r(batch_x, batch_y, MI_net, r):
    '''
    batch_x: data summaries
    batch_y: parameters
    MI_net: network for approximate f(x,y)
    r: network for approximate r(y|x)

    combine the input variable x batch [bs_size,feature_size]/variable y batch [bs_size,feature_size] 
    with samples from rinto score Matrix [bs_size,bs_size]
    where input (x,y) is treated as samples from p(x,y), insert into dignal of score matrix
    and r is treated as samples from r(x,y), insert into off-diagonal of score matrix
    see arXiv:2306.00608 for explanation
    '''
    batch_size = batch_x.size()[0]


    #add dim in batches so that we can combine input batches into a [bs_size,bs_size,feature_size] tensor
    batch_x_ = batch_x.view(batch_size, 1, -1).expand(-1, batch_size, -1)

    #sample batch_size pair of y for each x from r(y|x)
    cond_x=batch_x_.reshape(batch_size*batch_size,-1)
    cond_x=cond_x.to(batch_x.device)

    with torch.no_grad():
        batch_y_=(r.sample(batch_size**2,cond_inputs=cond_x)).view(batch_size,batch_size,-1)
        #set dignal elements to true y samples from p(x,y)
        batch_y_[range(batch_size), range(batch_size)] = batch_y

    score = MI_net(batch_x_, batch_y_)

    assert score.size()[-1] == 1, f"wrong score dim:{score.size()}"
    score_matrix = score.squeeze(-1)

    return score_matrix


def log_mean_exp_nodiag(score, dim=None):
    '''
    function for calculating logmeanexp, currently only support symetric (N*N) score
    '''
    bs = score.size()[0]
    device = score.device

    score_nodiag = score - torch.diag(
        float('inf') * torch.ones(bs, device=device))

    if dim is None:
        k = bs * (bs - 1) * torch.ones(1, device=device)
        logsumexp = torch.logsumexp(torch.flatten(score_nodiag), dim=0)
    else:
        k = (bs - 1) * torch.ones(1, device=device)
        logsumexp = torch.logsumexp(score_nodiag, dim=dim)

    return logsumexp - torch.log(k)


def tuba_lower_bound(score, log_baseline=0.):

    score = score - log_baseline
    joint_mean = torch.mean(torch.diag(score))
    marg_mean = torch.exp(log_mean_exp_nodiag(score))

    mi = 1. + joint_mean - marg_mean

    return mi


def nwj_lower_bound(score):
    return tuba_lower_bound(score, log_baseline=1.)


def infonce_lower_bound(score):

    bs = score.size()[0]

    nll = torch.mean(torch.diag(score) - torch.logsumexp(score, dim=1))

    mi = torch.log(bs) + nll

    return mi


def js_lower_bound(score):

    bs = score.size()[0]

    score_diag = torch.diag(score)
    first_term = torch.mean(-F.softplus(-score_diag))
    second_term = (torch.sum(F.softplus(score)) -
                   torch.sum(F.softplus(score_diag))) / (bs * (bs - 1))

    return first_term - second_term


def mi_js(score):

    return (js_lower_bound(score) +
            torch.log(4 * torch.ones(1, device=score.device))) / 2


def nwj_with_js(score):

    js_lb = js_lower_bound(score)
    mi = nwj_lower_bound(score.clone().detach())

    return js_lb + (mi - js_lb).detach()


def smile_lower_bound(f, clip=1):
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = log_mean_exp_nodiag(f_)
    dv = f.diag().mean() - z

    js = js_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js


def get_MI_estimator(name):
    if name == 'JS':
        return nwj_with_js
    elif name == 'nwj':
        return nwj_lower_bound
    elif name == 'smile':
        return smile_lower_bound
    else:
        assert False, 'MI estimator type not implemented'


'''
generative way of MI estimation
'''

# def MI_ba_lower_bound(log_condition,Hx,bs):
#     mi_ba=log_condition+Hx/bs
