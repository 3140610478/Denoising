import os
import sys
import torch

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.DnCNN import DnCNN as DnCNN
    from Networks import Training
    from Data import BSDS, REF
    from Log.Logger import getLogger
    import config

criterion = torch.nn.CrossEntropyLoss()


def run(model, data, name=None):
    if not isinstance(name, str):
        name = f"{model.__class__.__name__}_on_{data.__name__.rsplit('.')[-1]}"

    logger = getLogger(name)

    optimizers = Training.get_optimizers(
        model,
        (0.1, 0.01, 0.001, 0.0001)
    )

    logger.info(f"{name}\n")
    logger.info("learning_rate = 0.0001")
    best = Training.train_epoch_range(
        model, data, optimizers[0.0001], logger, 0, 50, best
    )
    Training.test(model, data, logger)


if __name__ == "__main__":
    run(DnCNN(REF.in_channels, config.layers).to(config.device), REF)
    run(DnCNN(BSDS.in_channels, config.layers).to(config.device), BSDS)
    pass
