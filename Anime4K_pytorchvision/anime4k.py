
# %%
# import numpy as np
import torch
import numpy as np

device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def show(*imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = to_pil_image(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# %%
from glob import glob
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose, ColorJitter
from torchvision.transforms.functional import resize
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, transform=None):
        self.imgs = glob("Dataset_1024/*.png")
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx])
        w, h = image.size
        if self.transform:
            image = self.transform(image)
        return resize(image, [w // 2, h // 2]), image

dataset = CustomImageDataset(Compose([
    ColorJitter(.4, .4, .4, .4), ToTensor(),
]))
train_dataset, test_dataset= random_split(dataset, [len(dataset) - 100, 100])

# %%
import torch.nn.functional as F
from torch import nn

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)

def create_model(model_layers = 8, hidden_channels = 8):
    layers = []
    layers.append(nn.Conv2d(3, hidden_channels, (3, 3), padding=1))
    layers.append(CReLU())
    for i in range(model_layers - 2):
        layers.append(nn.Conv2d(hidden_channels * 2, hidden_channels, (3, 3), padding=1))
        layers.append(CReLU())
    layers.append(nn.ConvTranspose2d(hidden_channels * 2, 3, (2, 2), stride=2))
    return nn.Sequential(*layers)

model = create_model().to(device)

# %%
def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return (y, cb, cr)

MSE_Loss = nn.MSELoss()
#YUV loss to weigh in favour of luminance (2 to 1), as humans are less sensitive to chroma degradation
def YUV_Error(y_true, y_pred):
    true_y, true_u, true_v = rgb_to_ycbcr(y_true)
    pred_y, pred_u, pred_v = rgb_to_ycbcr(y_pred)

    y_err = MSE_Loss(true_y, pred_y) * 0.5
    u_err = MSE_Loss(true_u, pred_u) * 0.25
    v_err = MSE_Loss(true_v, pred_v) * 0.25
    
    return (y_err + u_err + v_err)

# %%
from tqdm import tqdm

epochs=20
batch_size=32

optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
train_dataloader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#作用域闭包
def training_step(epoch, dataloader,model,loss_fn,optimizer):
    bar = tqdm(dataloader, f"Epoch {epoch}")
    for X, y in bar:
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.set_postfix_str(f"loss: {loss.item():>7f}")

def testing_step(epoch, dataloader,model,loss_fn,optimizer):
    bar = tqdm(dataloader, f"Test {epoch}")
    losses = []
    for X, y in bar:
        with torch.no_grad():
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))
            losses.append(loss.item())
        bar.set_postfix_str(f"loss: {loss.item():>7f}")
    print(f"loss: {torch.mean(torch.tensor(losses)).item():>7f}")

for t in range(epochs):
    training_step(t + 1, train_dataloader, model, MSE_Loss, optimizer)
    testing_step(t + 1, test_dataloader, model, MSE_Loss, optimizer)

print("Time end for training.")

# %%

def decorator(fn):
    print("decorator")
    def _fn():
        print("other")
        # fn()
    return _fn

@decorator
def test():
    print("test")

test()
# %%

def decorator(fn):
    print("decorator")
    def _fn():
        print("other")
        # fn()
    return _fn

def test():
    print("test")

test = decorator(test)

test()
# %%
