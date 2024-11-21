import matplotlib.pyplot as plt
import einops


def plot_image(image, ax=None):
    if ax is None:
        ax = plt.gca()
    image = einops.rearrange(image, "c h w -> h w c")
    ax.imshow(image.cpu().detach().numpy())