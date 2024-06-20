import os
import sys
import plotly.graph_objects as go
from plotly.express.colors import qualitative as colorset
from plotly.offline import plot
from plotly.subplots import make_subplots
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
log_folder = os.path.abspath(os.path.join(base_folder, "./Log/"))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Networks.DnCNN import DnCNN
    from Data import BSDS, REF


def read_training_log(filename):
    with open(os.path.abspath(os.path.join(log_folder, f"./{filename}"))) as log:
        lines = log.readlines()
    best_epoch = int(lines[-1].strip().split()[-1])
    lines = [i.split("\t") for i in lines[4:-2] if i != "\n"]
    lines = [[j.rstrip().rstrip(",") for j in i] for i in lines if len(i) > 1]
    groups = ("train", "val", "test",)
    contents = {
        "[loss]": {_: [] for _ in groups},
        "[psnr]": {_: [] for _ in groups},
        "[ssim]": {_: [] for _ in groups},
        "[best_epoch]": [best_epoch],
    }
    for line in lines:
        content = contents[line[0]]
        for record in line[1:]:
            group, value = record.split(":")
            content[group].append(float(value))
    contents["[len]"] = len(contents["[loss]"]["train"])
    return contents


def plot_contents(fig: go.Figure, name: str, contents: dict[str, dict[str, list]], colors: list[str]) -> None:
    length, loss, psnr, ssim = contents["[len]"], contents["[loss]"], contents["[psnr]"], contents["[ssim]"]
    best_epoch = contents["[best_epoch]"]
    x = list(range(1, 1+length))
    fig.add_trace(
        go.Scatter(
            name=name+" train",
            x=x,
            y=loss["train"],
            mode="lines",
            line={"dash": "dot", "color": colors[0], },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" val",
            x=x,
            y=loss["val"],
            mode="lines",
            line={"color": colors[1], },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" test",
            x=best_epoch,
            y=loss["test"],
            mode="markers",
            marker={"color": colors[2], }
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" train",
            x=x,
            y=psnr["train"],
            mode="lines",
            line={"dash": "dot", "color": colors[0], },
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" val",
            x=x,
            y=psnr["val"],
            mode="lines",
            line={"color": colors[1], },
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" test",
            x=best_epoch,
            y=psnr["test"],
            mode="markers",
            marker={"color": colors[2], }
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" train",
            x=x,
            y=ssim["train"],
            mode="lines",
            line={"dash": "dot", "color": colors[0], },
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" val",
            x=x,
            y=ssim["val"],
            mode="lines",
            line={"color": colors[1], },
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            name=name+" test",
            x=best_epoch,
            y=ssim["test"],
            mode="markers",
            marker={"color": colors[2], }
        ),
        row=1,
        col=3,
    )
    psnr_max = max(psnr["train"]+psnr["val"]+psnr["test"])
    fig.update_xaxes(title_text="epochs", row=1, col=1)
    fig.update_xaxes(title_text="epochs", row=1, col=2)
    fig.update_xaxes(title_text="epochs", row=1, col=3)
    fig.update_yaxes(title_text="loss", range=(0, 0.5), row=1, col=1)
    fig.update_yaxes(title_text="psnr", range=(0, psnr_max+1), row=1, col=2)
    fig.update_yaxes(title_text="ssim", range=(0, 1.1), row=1, col=3)
    pass


if __name__ == '__main__':
    folder = base_folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    model, dataset = DnCNN, REF
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("[loss]", "[psnr]", "[ssim]"),
        horizontal_spacing=0.05,
    )
    colors = colorset.Plotly
    name = name = f"{model.__name__}_on_{dataset.__name__.rsplit('.')[-1]}"
    log = read_training_log(f"./{name}.log")

    plot_contents(fig, f"{model.__name__}", log, colors)

    plot(fig, filename=os.path.abspath(
        os.path.join(base_folder, f"./{name}.html")))
    
    model, dataset = DnCNN, BSDS
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("[loss]", "[psnr]", "[ssim]"),
        horizontal_spacing=0.05,
    )
    colors = colorset.Plotly
    name = name = f"{model.__name__}_on_{dataset.__name__.rsplit('.')[-1]}"
    log = read_training_log(f"./{name}.log")

    plot_contents(fig, f"{model.__name__}", log, colors)

    plot(fig, filename=os.path.abspath(
        os.path.join(base_folder, f"./{name}.html")))
    pass
