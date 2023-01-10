import math

import torch
import torch.nn as nn
import torch.nn.init as init
from einops import repeat

from . import pmath


class HyperbolicMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, ball_dim, n_classes, c):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = pmath.expmap0(self.p_vals, c=c)
        # p_vals_poincare = self.p_vals
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = pmath._hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes, self.c
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))


class HypClassifer(nn.Module):
    def __init__(self, in_features, num_classes, c, bias=True):
        super(HypClassifer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        # self.c = nn.Parameter(torch.tensor(c))
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(self.num_classes, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_classes,self.in_features))
            self.bias_exp = nn.Parameter(torch.Tensor(self.num_classes))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.bias_exp, -bound, bound)

    def forward(self, x):
        #Rearranging The shape of tensors
        weight = repeat(self.weight,"c f-> b c f", b =x.shape[0])
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        b_norm = torch.norm(self.bias, dim=-1, keepdim=True)
        w_norm = torch.norm(self.weight, dim=-1, keepdim=True)
        x = repeat(x,"b f -> b c f", c=self.num_classes)
        bias = repeat(self.bias,"c f-> b c f", b=x.shape[0])
        x_norm = repeat(x_norm,"b o -> b c o", c=self.num_classes)
        b_norm = repeat(b_norm,"c o -> b c o", b=x.shape[0]).detach()
        w_norm = repeat(w_norm,"c o -> b c o", b=x.shape[0]).detach()
        #bx = torch.sum(torch.mul(x,b_norm),dim=-1, keepdim=True).detach()
        weight = weight / w_norm
        """ Calculate angle between bias and x
        """
        ang = (x*bias)/(x_norm * b_norm)
        abs_ang = torch.abs(ang)
        sign_angle = torch.sign(ang)
        p_idx = (sign_angle + 1) / 2
        n_idx = (-sign_angle + 1) / 2

        ratio = x_norm / b_norm

        negsign = torch.sign(-3*ratio +1)

        pos_weight = (torch.pow(3*ratio+1,1./3) - 1)
        neg_weight = -(negsign*torch.pow(torch.abs(-3*ratio + 1),1./3) + 1)
        zero_weight = (torch.pow(torch.sqrt(9*ratio**2+4) - 3*ratio, 1./3) / 2**(1/3))

        x = bias +  (p_idx*pos_weight*abs_ang+n_idx*neg_weight*abs_ang+zero_weight*(1-abs_ang))*x/b_norm
        #x = (x_norm*x + b_norm*bias)/(x_norm+b_norm)
        #x *= torch.norm(x,dim=-1,keepdim=True).detach()
        #x = x + (bx*bias)/10
        x = torch.mul(x,weight)
        #x *= x_norm
        logits = torch.sum(x, dim=-1, keepdim=False)
        #logits = logits * w_norm.T

        #logits = (x @ self.weight.T) + self.bias
        #logits = logits + x_norm
        #sign = torch.sign(logits)
        #logits = torch.log(torch.abs(logits) + 1)
        #x_norm = torch.log(x_norm + 1)
        #logits = torch.abs(logits) + x_norm
        #logits = sign * logits
        #logits *= w_norm.T
        # logits = logits * x_norm
        # logits = x + self.bias
        # logits = -x * norm + self.bias
        return logits, logits

class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


class ConcatPoincareLayer(nn.Module):
    def __init__(self, d1, d2, d_out, c):
        super(ConcatPoincareLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.l1 = HypLinear(d1, d_out, bias=False, c=c)
        self.l2 = HypLinear(d2, d_out, bias=False, c=c)
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.mobius_add(self.l1(x1), self.l2(x2), c=c)

    def extra_repr(self):
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return "c={}".format(self.c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c, ]))
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c

        self.clip_r = clip_r

        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        x_norm = 0
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(
                torch.ones_like(x_norm),
                self.clip_r / x_norm
            )
            x = x * fac

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c)), x_norm
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c)), x_norm

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c, ]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.logmap(xp, x, c=self.c)
        return pmath.logmap0(x, c=self.c)

    def extra_repr(self):
        return "train_c={}, train_x={}".format(self.train_c, self.train_x)

