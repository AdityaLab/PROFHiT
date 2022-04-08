from importlib.metadata import metadata
import numpy as np
import pandas as pd
import torch as th
from torch.nn.functional import mse_loss
from hierarchy_data import (
    LabourHierarchyData,
    TourismHierarchyData,
    WikiHierarchyData,
    normalize_data,
    unnormalize_data,
)
from models.fnpmodels import EmbedMetaAttenSeq, RegressionSepFNP, Corem
from utils import lag_dataset
from models.utils import float_tensor, long_tensor
from tqdm import tqdm
import properscoring as ps

SEED = 42
DEVICE = "cuda"
DATASET = "TOURISM"
AHEAD = 12
TRAIN_UPTO = 228 - 12
BACKUP_TIME = 30
PRE_BATCH_SIZE = 10
PRE_TRAIN_LR = 0.001
PRE_TRAIN_EPOCHS = 10
FRAC_VAL = 0.1
C = 5.0
BATCH_SIZE = 10
TRAIN_LR = 0.001
LAMBDA = 0.1
TRAIN_EPOCHS = 100
EVAL_SAMPLES = 100

np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)

if DEVICE == "cuda":
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
else:
    device = th.device("cpu")


data_obj = TourismHierarchyData()

# data_obj = normalize_data(data_obj)


# Let's create dataset
full_data = data_obj.data
train_data_raw = full_data[:, :TRAIN_UPTO]
train_means = np.mean(train_data_raw, axis=1)
train_std = np.std(train_data_raw, axis=1)
train_data = (train_data_raw - train_means[:, None]) / train_std[:, None]
# train_data = train_data_raw

dataset_raw = lag_dataset(train_data, BACKUP_TIME)


class SeqDataset(th.utils.data.Dataset):
    def __init__(self, dataset):
        self.X, self.Y = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dataset[idx]


dataset = SeqDataset(dataset_raw)


# Let's create FNP model
encoder = EmbedMetaAttenSeq(
    dim_seq_in=1,
    num_metadata=data_obj.data.shape[0],
    dim_metadata=1,
    dim_out=60,
    n_layers=2,
    bidirectional=True,
).to(device)
decoder = RegressionSepFNP(
    dim_x=60,
    dim_y=1,
    dim_h=60,
    n_layers=3,
    dim_u=60,
    dim_z=60,
    nodes=data_obj.data.shape[0],
).to(device)


pre_opt = th.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=PRE_TRAIN_LR
)

# Create training validation set


perm = np.random.permutation(np.arange(BACKUP_TIME, TRAIN_UPTO))
train_idx = perm[: int(len(perm) * (1 - FRAC_VAL))]
val_idx = perm[int(len(perm) * (1 - FRAC_VAL)) :]


def pretrain_epoch():
    encoder.train()
    decoder.train()
    losses = []
    means, stds = [], []
    pre_opt.zero_grad()
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(train_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample, logstd_sample, log_py, log_pqz, _ = decoder(ref_out_x, out_x, y)
        loss = -(log_py + log_pqz) / x.shape[0]
        loss.backward()
        losses.append(loss.detach().cpu().item())
        means.append(mean_sample.detach().cpu().numpy())
        stds.append(logstd_sample.detach().cpu().numpy())
        if (i + 1) % PRE_BATCH_SIZE == 0:
            pre_opt.step()
            pre_opt.zero_grad()
    if i % PRE_BATCH_SIZE != 0:
        pre_opt.step()
    return np.mean(losses), np.array(means), np.array(stds)


def pre_validate(sample=False):
    encoder.eval()
    decoder.eval()
    losses = []
    means, stds = [], []
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(val_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=sample)
        mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
        losses.append(mse_loss)
        means.append(mean_y.detach().cpu().numpy())
        stds.append(logstd_y.detach().cpu().numpy())
    return np.mean(losses), np.array(means), np.array(stds)


print("Pretraining...")
for ep in tqdm(range(PRE_TRAIN_EPOCHS)):
    loss, means, stds = pretrain_epoch()
    print(f"Epoch {ep} loss: {loss}")
    with th.no_grad():
        loss, means, stds = pre_validate()
        print(f"Epoch {ep} Val loss: {loss}")

# Let's real_train

corem = Corem(nodes=data_obj.data.shape[0], c=C,).to(device)

opt = th.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(corem.parameters()),
    lr=TRAIN_LR,
)


def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2 - 1.0


def generate_hmatrix():
    ans1 = np.zeros(data_obj.data.shape[0], data_obj.data.shape[0])
    for i, n in enumerate(data_obj.nodes1):
        if len(n.children) == 0:
            ans1[n.idx, n.idx] = 1
        c_idx = [x.idx for x in n.children]
        ans1[n.idx, c_idx] = 1.0
    ans2 = np.zeros(data_obj.data.shape[0], data_obj.data.shape[0])
    for i, n in enumerate(data_obj.nodes2):
        if len(n.children) == 0:
            ans2[n.idx, n.idx] = 1
        c_idx = [x.idx for x in n.children]
        ans2[n.idx, c_idx] = 1.0
    return float_tensor(ans1), float_tensor(ans2)


def jsd_loss(mu, logstd, hmatrix, train_means, train_std):
    lhs_mu = (((mu * train_std + train_means) * hmatrix).sum(1) - train_means) / (
        train_std
    )
    lhs_var = (((th.exp(2.0 * logstd) * (train_std ** 2)) * hmatrix).sum(1)) / (
        train_std ** 2
    )
    ans = th.nan_to_num(jsd_norm(mu, lhs_mu, (2.0 * logstd).exp(), lhs_var))
    return ans.mean()


def train_epoch():
    encoder.train()
    decoder.train()
    corem.train()
    losses = []
    means, stds, gts = [], [], []
    opt.zero_grad()
    ref_x = float_tensor(train_data[:, :, None])
    hmatrix1, hmatrix2 = generate_hmatrix()
    th_means = float_tensor(train_means)
    th_std = float_tensor(train_std)
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(train_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample1, logstd_sample1, log_py1, log_pqz, py1 = decoder(
            ref_out_x, out_x, y
        )
        mean_sample, logstd_sample, log_py, py = corem(
            mean_sample1.squeeze(), logstd_sample1.squeeze(), y
        )
        loss1 = -(log_py + log_pqz) / x.shape[0]
        loss2 = (
            jsd_loss(
                mean_sample.squeeze(),
                logstd_sample.squeeze(),
                hmatrix1,
                th_means,
                th_std,
            )
            / x.shape[0]
        )
        loss3 = (
            jsd_loss(
                mean_sample.squeeze(),
                logstd_sample.squeeze(),
                hmatrix2,
                th_means,
                th_std,
            )
            / x.shape[0]
        )
        loss = loss1 + LAMBDA * (loss2 + loss3)
        if th.isnan(loss):
            import pdb

            pdb.set_trace()
        loss.backward()
        losses.append(loss.detach().cpu().item())
        print(f"Loss1: {loss1.detach().cpu().item()}")
        print(f"Loss2: {loss2.detach().cpu().item()}")
        means.append(mean_sample.detach().cpu().numpy())
        stds.append(logstd_sample.detach().cpu().numpy())
        gts.append(y.detach().cpu().numpy())
        if (i + 1) % BATCH_SIZE == 0:
            opt.step()
            opt.zero_grad()
    if i % BATCH_SIZE != 0:
        opt.step()
    return np.mean(losses), np.array(means), np.array(stds)


def validate(sample=False):
    encoder.eval()
    decoder.eval()
    corem.eval()
    losses = []
    means, stds, gts = [], [], []
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(val_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
        y_pred, mean_y, logstd_y, _ = corem.predict(
            mean_y.squeeze(), logstd_y.squeeze(), sample=sample
        )
        mse_loss = np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2)
        losses.append(mse_loss)
        means.append(mean_y.detach().cpu().numpy())
        stds.append(logstd_y.detach().cpu().numpy())
        gts.append(y.cpu().numpy())
    return np.mean(losses), np.array(means), np.array(stds)


print("Training....")
for ep in tqdm(range(TRAIN_EPOCHS)):
    loss, means, stds = train_epoch()
    print(f"Epoch {ep} loss: {loss}")
    with th.no_grad():
        loss, means, stds = validate()
        print(f"Epoch {ep} Val loss: {loss}")

# Lets evaluate

# One sampple
def sample_data():
    curr_data = train_data.copy()
    encoder.eval()
    decoder.eval()
    corem.eval()
    for t in range(AHEAD):
        ref_x = float_tensor(train_data[:, :, None])
        meta_x = long_tensor(np.arange(ref_x.shape[0]))
        x = float_tensor(curr_data[:, :, None])
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
        y_pred, mean_y, logstd_y, _ = corem.predict(
            mean_y.squeeze(), logstd_y.squeeze(), sample=True
        )
        y_pred = y_pred.cpu().numpy()
        curr_data = np.concatenate([curr_data, y_pred], axis=1)
    return curr_data[:, -AHEAD:]


ground_truth = full_data[:, TRAIN_UPTO : TRAIN_UPTO + AHEAD]
with th.no_grad():
    preds = [sample_data() for _ in tqdm(range(EVAL_SAMPLES))]
preds = np.array(preds)
preds = preds * train_std[:, None] + train_means[:, None]
mean_preds = np.mean(preds, axis=0)
rmse = np.sqrt(np.mean((ground_truth - mean_preds) ** 2))
print(f"RMSE: {rmse}")
crps = ps.crps_ensemble(ground_truth, np.moveaxis(preds, [1, 2, 0], [0, 1, 2])).mean()
print(f"CRPS: {crps}")
