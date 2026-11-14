"""test_attacks.py - Attack the trained MNIST model on the *test* split.

Metrics
-------
* Recognition rate (clean): fraction of test images classified correctly, no attack.
* Accuracy under attack:    fraction still classified correctly after the attack.
* ASR (attack success rate): among images the model originally got RIGHT,
                             the fraction it gets WRONG after the attack.
Attacks are untargeted and L-infinity bounded: no pixel moves by more than eps.
"""
import csv
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import Net, DEVICE

EPSILONS = [0.05, 0.1, 0.2, 0.3]
STEPS = 10      # iterations for I-FGSM / MI-FGSM
MU = 1.0        # momentum decay for MI-FGSM


def grad_of_loss(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    return torch.autograd.grad(loss, x)[0]


def fgsm(model, x, y, eps):
    """One step of size eps in the direction that increases the loss."""
    g = grad_of_loss(model, x, y)
    return (x + eps * g.sign()).clamp(0, 1).detach()


def ifgsm(model, x, y, eps, steps=STEPS):
    """Iterative FGSM (a.k.a. PGD without random start): many small steps, projected back into eps-ball."""
    alpha = eps / steps
    x_adv = x.clone().detach()
    for _ in range(steps):
        g = grad_of_loss(model, x_adv, y)
        x_adv = x_adv + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


def pgd(model, x, y, eps, steps=STEPS):
    """PGD: I-FGSM with a random start inside the eps-ball."""
    alpha = 2.5 * eps / steps
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        g = grad_of_loss(model, x_adv, y)
        x_adv = x_adv + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


def mifgsm(model, x, y, eps, steps=STEPS, mu=MU):
    """Momentum I-FGSM: accumulates a running average of gradients to stabilise the direction."""
    alpha = eps / steps
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        g = grad_of_loss(model, x_adv, y)
        g = g / g.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        momentum = mu * momentum + g
        x_adv = x_adv + alpha * momentum.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


ATTACKS = {"FGSM": fgsm, "I-FGSM": ifgsm, "PGD": pgd, "MI-FGSM": mifgsm}


def main():
    torch.manual_seed(0)
    model = Net().to(DEVICE)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=DEVICE))
    model.eval()

    test_set = datasets.MNIST("./data", train=False, download=True,
                              transform=transforms.ToTensor())
    loader = DataLoader(test_set, batch_size=500)
    n_total = len(test_set)

    n_clean_correct = 0
    stats = {(a, e): {"still_correct": 0, "fooled": 0} for a in ATTACKS for e in EPSILONS}

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            clean_ok = model(x).argmax(1) == y
        n_clean_correct += clean_ok.sum().item()

        for name, fn in ATTACKS.items():
            for eps in EPSILONS:
                x_adv = fn(model, x, y, eps)
                with torch.no_grad():
                    adv_ok = model(x_adv).argmax(1) == y
                stats[(name, eps)]["still_correct"] += adv_ok.sum().item()
                stats[(name, eps)]["fooled"] += (clean_ok & ~adv_ok).sum().item()

    clean_rate = n_clean_correct / n_total
    rows = [("attack", "epsilon", "clean_recognition_rate", "accuracy_under_attack", "ASR")]
    print(f"Clean recognition rate: {clean_rate:.4f}\n")
    print(f"{'Attack':8s} {'eps':>5s} {'Acc under attack':>17s} {'ASR':>8s}")
    for (name, eps), s in stats.items():
        acc = s["still_correct"] / n_total
        asr = s["fooled"] / n_clean_correct
        rows.append((name, eps, f"{clean_rate:.4f}", f"{acc:.4f}", f"{asr:.4f}"))
        print(f"{name:8s} {eps:5.2f} {acc:17.4f} {asr:8.4f}")

    with open("results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\nSaved results.csv")


if __name__ == "__main__":
    main()
