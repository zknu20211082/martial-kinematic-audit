"""Compact ST-GCN (Yan et al. 2018 style) for skeleton sequences, plus train/eval helpers."""
import math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def build_adjacency(n_joints, bones, root=0):
    """3-subset spatial partition: identity / centripetal (closer to root) / centrifugal."""
    import collections
    adj = collections.defaultdict(list)
    for a, b in bones:
        adj[a].append(b); adj[b].append(a)
    hop = np.full(n_joints, 1e9); hop[root] = 0; q = [root]
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if hop[v] > hop[u] + 1:
                hop[v] = hop[u] + 1; q.append(v)
    A = np.zeros((3, n_joints, n_joints), np.float32)
    for i in range(n_joints):
        A[0, i, i] = 1
    for a, b in bones:
        for u, v in ((a, b), (b, a)):
            if hop[v] < hop[u]:
                A[1, u, v] = 1   # v is closer to root (centripetal)
            else:
                A[2, u, v] = 1
    for k in range(3):   # symmetric normalisation
        d = A[k].sum(1, keepdims=True); d[d == 0] = 1
        A[k] = A[k] / d
    return torch.tensor(A)


class GCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, tk=9, dropout=0.2, residual=True):
        super().__init__()
        self.register_buffer("A", A)
        self.K = A.shape[0]
        self.gcn = nn.Conv2d(cin, cout * self.K, 1)
        self.edge = nn.Parameter(torch.ones_like(A))
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, (tk, 1), (stride, 1), ((tk - 1) // 2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(dropout))
        if not residual:
            self.res = lambda x: 0
        elif cin == cout and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))

    def forward(self, x):  # x: N,C,T,V
        r = self.res(x)
        n, c, t, v = x.shape
        y = self.gcn(x).view(n, self.K, -1, t, v)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A * self.edge)
        return F.relu(self.tcn(y) + r)


class STGCN(nn.Module):
    def __init__(self, in_ch, n_joints, bones, n_out, n_persons=1, base=64, dropout=0.2):
        super().__init__()
        A = build_adjacency(n_joints, bones)
        self.n_persons = n_persons
        self.bn = nn.BatchNorm1d(in_ch * n_joints)
        chans = [(in_ch, base, 1, False), (base, base, 1, True), (base, base, 1, True), (base, 2 * base, 2, True),
                 (2 * base, 2 * base, 1, True), (2 * base, 4 * base, 2, True), (4 * base, 4 * base, 1, True)]
        self.blocks = nn.ModuleList([GCNBlock(ci, co, A, s, dropout=dropout, residual=r) for ci, co, s, r in chans])
        self.fc = nn.Linear(4 * base, n_out)

    def forward(self, x):  # x: N,C,T,V,M
        n, c, t, v, m = x.shape
        x = x.permute(0, 4, 3, 1, 2).reshape(n * m, v * c, t)
        x = self.bn(x).view(n * m, v, c, t).permute(0, 2, 3, 1)
        for b in self.blocks:
            x = b(x)
        x = x.mean(dim=(2, 3)).view(n, m, -1).mean(1)
        return self.fc(x)


def augment3d(X, rng):
    """X: (N,T,V,3) height-normalised, pelvis-centred. Random yaw rotation, scale, temporal crop-resize, jitter."""
    N, T, V, _ = X.shape
    out = X.copy()
    for i in range(N):
        th = rng.uniform(-math.pi, math.pi)
        R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]], np.float32)
        s = rng.uniform(0.9, 1.1)
        seq = (out[i] @ R.T) * s
        # temporal crop 80-100% then resize back
        L = int(T * rng.uniform(0.8, 1.0)); st = rng.integers(0, T - L + 1)
        idx = np.linspace(st, st + L - 1, T).round().astype(int)
        seq = seq[idx]
        seq += rng.normal(0, 0.005, seq.shape).astype(np.float32)
        out[i] = seq
    return out


def to_tensor3d(X):  # (N,T,V,3) -> (N,3,T,V,1)
    return torch.tensor(np.transpose(X, (0, 3, 1, 2))[..., None], dtype=torch.float32)


def train_model(model, Xtr, ytr, Xva=None, yva=None, task="reg", epochs=40, bs=32, lr=1e-3, aug=None, seed=0, device="cuda", verbose=False):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(Xtr) / bs))
    ytr_t = torch.tensor(ytr, dtype=torch.float32 if task == "reg" else torch.long)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(Xtr)); tot = 0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            xb = Xtr[b]
            if aug is not None:
                xb = aug(xb, rng)
            xb = xb.to(device) if torch.is_tensor(xb) else to_tensor3d(xb).to(device)
            out = model(xb)
            if task == "reg":
                loss = F.smooth_l1_loss(out.squeeze(1), ytr_t[b].to(device))
            else:
                loss = F.cross_entropy(out, ytr_t[b].to(device), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            tot += loss.item() * len(b)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ep{ep} loss {tot / len(Xtr):.4f}")
    return model


@torch.no_grad()
def predict(model, X, bs=64, device="cuda"):
    model.eval(); outs = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        xb = xb.to(device) if torch.is_tensor(xb) else to_tensor3d(xb).to(device)
        outs.append(model(xb).float().cpu().numpy())
    return np.concatenate(outs)
