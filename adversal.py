import torch
import torch.nn as nn

def fgsm(model, x, target, eps, targeted=True):
  x.requires_grad_()
  L = nn.CrossEntropyLoss()
  eps_new = eps - 1e-7
  loss = L(model(x), target)
  loss.backward()
  if targeted:
    x = x - eps_new * x.grad.sign()
  else:
    x = x + eps_new * x.grad.sign()
  x = x.clamp(*(0, 1))
  return x


def pgd_untargeted(model, x, labels, eps, eps_step, num_itr=30):
  x_new = x.clone().requires_grad_(True)
  L = nn.CrossEntropyLoss()

  for i in range(num_itr):
    x_new = x_new.clone().detach().requires_grad_(True)
    loss = L(model(x_new), labels)
    loss.backward()
    with torch.no_grad():
      x_new = x_new + x_new.grad.sign() * eps_step
    x_new = torch.clamp(x_new, x - eps, x + eps)
    x_new = x_new.clamp(*(0, 1))
  return x_new.detach()
