import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from logging import Logger
from typing import Iterable
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


class PSNR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, eps=1e-5) -> torch.Tensor:
        mse = F.mse_loss(input, target)
        psnr = - 10 * torch.log10(mse)
        return psnr


class SSIM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, eps=1e-5) -> torch.Tensor:
        x, y = input.flatten(), target.flatten()
        mu_x, mu_y = x.mean(), y.mean()
        sigma_xx, sigma_xy, sigma_yx, sigma_yy = \
            torch.cov(torch.stack((x, y))).flatten()
        ssim = \
            ((2*mu_x*mu_y + eps) * (2*sigma_xy + eps)) / \
            ((mu_x**2 + mu_y**2 + eps) * (sigma_xx + sigma_yy + eps))
        return ssim


psnr_fun = PSNR()
ssim_fun = SSIM()


def get_optimizers(
    model: torch.nn.Module,
    learning_rate: Iterable[float] = (0.1, 0.01, 0.001, 0.0001)
):
    return {
        # lr: torch.optim.SGD(
        #     model.parameters(),
        #     lr=lr,
        #     momentum=0.9,
        #     weight_decay=0.0001,
        # )
        lr: torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=0.0001,
        )
        for lr in learning_rate
    }


def train_epoch(
    model: torch.nn.Module,
    data,
    logger: Logger,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    best: float = 0,
) -> tuple[float]:
    message = \
        "[loss]\ttrain:{:.8f},\tval:{:.8f}\n" + \
        "[psnr]\ttrain:{:.8f},\tval:{:.8f}\n" + \
        "[ssim]\ttrain:{:.8f},\tval:{:.8f}\n"

    logger.info("\n[Epoch {:0>4d}]".format(epoch+1))
    train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim = 0, 0, 0, 0, 0, 0

    model.train()
    print("\nTraining:")
    for sample in tqdm(data.train_loader):
        x, y = sample
        x, y = x.to(config.device), y.to(config.device)

        optimizer.zero_grad()

        h = x - model(x)
        loss = F.mse_loss(h, y)
        ssim = ssim_fun(h, y)
        psnr = psnr_fun(h, y)
        loss.backward()
        optimizer.step()

        train_loss += len(y)*(float(loss))
        train_psnr += len(y)*(float(psnr))
        train_ssim += len(y)*(float(ssim))
    train_loss /= data.len_train
    train_psnr /= data.len_train
    train_ssim /= data.len_train

    model.eval()
    print("\nValidating:")
    with torch.no_grad():
        for sample in tqdm(data.val_loader):
            x, y = sample
            x, y = x.to(config.device), y.to(config.device)

            h = x - model(x)
            loss = F.mse_loss(h, y)
            ssim = ssim_fun(h, y)
            psnr = psnr_fun(h, y)

            val_loss += len(y)*(float(loss))
            val_psnr += len(y)*(float(psnr))
            val_ssim += len(y)*(float(ssim))
        val_loss /= data.len_val
        val_psnr /= data.len_val
        val_ssim /= data.len_val

    if val_psnr > best:
        torch.save(
            {
                "epoch": epoch,
                "best": best,
                "state_dict": model.state_dict(),
            },
            config.save_path,
        )

    best = max(best, float(val_psnr))
    result = train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim, best
    print("")
    logger.info(message.format(*result))

    return result


def train_epoch_range(
    model: torch.nn.Module,
    data,
    logger: Logger,
    start: int,
    stop: int,
    optimizer: torch.optim.Optimizer,
    best=0,
) -> None:
    for epoch in trange(start, stop):
        best = train_epoch(model, data, logger, epoch, optimizer, best)[-1]
    return best


def train_until(
    model: torch.nn.Module,
    data,
    logger: Logger,
    threshold: float,
    optimizer: torch.optim.Optimizer,
    best=0,
) -> tuple[float, int]:
    epoch = 0
    train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim, best = \
        (0, 0, 0, 0, 0, 0, best)
    while train_psnr <= threshold or val_psnr <= threshold:
        train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim, best = \
            train_epoch(model, data, logger, epoch, optimizer, best)
        epoch += 1
    return best, epoch


@torch.no_grad()
def test(
    model: torch.nn.Module,
    data,
    logger: Logger,
) -> None:
    logger.info("\nTesting: ")

    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint["state_dict"])

    message = "[loss]\ttest:{:.8f}\n[psnr]\ttest:{:.8f}\n[ssim]\ttest:{:.8f}\n"
    test_loss, test_psnr, test_ssim = 0, 0, 0

    model.eval()
    print("\nTesting:")
    for sample in tqdm(data.test_loader):
        x, y = sample
        x, y = x.to(config.device), y.to(config.device)

        h = x - model(x)
        loss = F.mse_loss(h, y)
        ssim = ssim_fun(h, y)
        psnr = psnr_fun(h, y)

        test_loss += len(y)*(float(loss))
        test_psnr += len(y)*(float(psnr))
        test_ssim += len(y)*(float(ssim))
    test_loss /= data.len_test
    test_psnr /= data.len_test
    test_ssim /= data.len_test

    print("")
    logger.info(message.format(test_loss, test_psnr, test_ssim))
    logger.info(f"Best Epoch: {checkpoint['epoch']+1}")

    return None
