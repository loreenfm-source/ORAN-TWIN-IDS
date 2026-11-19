#!/usr/bin/env python3
"""EXPERIMENTAL - simplified TimeGAN generator for synthetic windows.

Not used for the released dataset (see ``experimental/README.md`` and
supplementary Section S1.3). Kept so that the evaluated generator can be
inspected and re-run.

Architecture: GRU embedder, recovery, generator and supervisor (2 layers;
supervisor 1 layer; sigmoid outputs in latent space) and a GRU discriminator
returning logits (``BCEWithLogitsLoss``). Training streams windows through a
reservoir shuffle buffer (20,000) in three phases:

1. autoencoder (embedder + recovery, MSE), ``--ae_epochs``;
2. supervisor next-step prediction on real latents, ``--sup_epochs``;
3. joint training, ``--gan_epochs``: generator loss
   ``L_U + gamma * sqrt(L_S) + eta * L_V`` where ``L_V`` matches the batch mean,
   the across-window std and the within-window std of the windows; embedder
   loss ``10 * sqrt(L_R) + 0.1 * L_S``; discriminator updated only while its
   loss exceeds 0.15.

Generation runs ``recovery(supervisor(generator(noise)))``; the constant MGEN
columns f19-f24 are set to 0. Afterwards, the within-window standard deviation
of the five injection targets is compared with a random sample of real
windows drawn from the whole archive (all clusters).

Outputs: ``generated_windows_scaled.npy`` (float32, (n, T, 25)) and
``timegan_weights.pth`` in ``--output_dir``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran_twin_ids.features import INJECTION_TARGETS, MGEN_FEATURES, N_FEATURES  # noqa: E402
from oran_twin_ids.variant import open_windows  # noqa: E402

log = logging.getLogger("timegan")
MGEN_SLICE = slice(N_FEATURES - len(MGEN_FEATURES), N_FEATURES)  # f19..f24
REFERENCE_SAMPLE = 20_000


class WindowsDataset(IterableDataset):
    """Streams windows from a memory-mapped ``.npy`` through a reservoir shuffle buffer."""

    def __init__(self, npy_path: Path, shuffle_buffer: int = 20_000, seed: int = 42):
        self.npy_path, self.shuffle_buffer, self.seed = npy_path, shuffle_buffer, seed

    def __iter__(self):
        windows = np.load(self.npy_path, mmap_mode="r")
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        indices = range(len(windows)) if info is None else range(wid, len(windows), info.num_workers)
        rng = np.random.RandomState(self.seed + wid)
        buf = []
        for i in indices:
            buf.append(windows[i].astype(np.float32))
            if len(buf) >= self.shuffle_buffer:
                j = rng.randint(len(buf))
                buf[j], buf[-1] = buf[-1], buf[j]
                yield buf.pop()
        rng.shuffle(buf)
        yield from buf


class GRUBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=n_layers, batch_first=True)

    def forward(self, x):
        return self.gru(x)[0]


class Embedder(nn.Module):
    def __init__(self, in_dim, hidden, latent):
        super().__init__()
        self.rnn, self.fc = GRUBlock(in_dim, hidden), nn.Linear(hidden, latent)

    def forward(self, x):
        return torch.sigmoid(self.fc(self.rnn(x)))


class Recovery(nn.Module):
    def __init__(self, latent, hidden, out_dim):
        super().__init__()
        self.rnn, self.fc = GRUBlock(latent, hidden), nn.Linear(hidden, out_dim)

    def forward(self, z):
        return self.fc(self.rnn(z))


class Generator(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.rnn, self.fc = GRUBlock(latent, hidden), nn.Linear(hidden, latent)

    def forward(self, noise):
        return torch.sigmoid(self.fc(self.rnn(noise)))


class Supervisor(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.rnn, self.fc = GRUBlock(latent, hidden, n_layers=1), nn.Linear(hidden, latent)

    def forward(self, h):
        return torch.sigmoid(self.fc(self.rnn(h)))


class Discriminator(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.rnn, self.fc = GRUBlock(latent, hidden), nn.Linear(hidden, 1)

    def forward(self, h):
        return self.fc(self.rnn(h)).mean(dim=1)  # logits


def train_timegan(npy_path: Path, data: np.ndarray, output_dir: Path, *, batch_size=512,
                  ae_epochs=10, sup_epochs=10, gan_epochs=50, latent_dim=16, hidden_dim=64,
                  gamma=1.0, eta=10.0, num_synth=150_000, seed=42) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, t, f = data.shape
    log.info("Device %s | data %s", device, data.shape)
    loader = DataLoader(WindowsDataset(npy_path, seed=seed), batch_size=batch_size, num_workers=0)

    embedder = Embedder(f, hidden_dim, latent_dim).to(device)
    recovery = Recovery(latent_dim, hidden_dim, f).to(device)
    generator = Generator(latent_dim, hidden_dim).to(device)
    supervisor = Supervisor(latent_dim, hidden_dim).to(device)
    discriminator = Discriminator(latent_dim, hidden_dim).to(device)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    opt_ae = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=1e-3)
    opt_sup = torch.optim.Adam(supervisor.parameters(), lr=1e-3)
    opt_g = torch.optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=1e-4)
    opt_e = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    for epoch in range(ae_epochs):  # phase 1
        for batch in loader:
            batch = batch.to(device)
            loss = mse(recovery(embedder(batch)), batch)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        log.info("AE %d/%d recon %.6f", epoch + 1, ae_epochs, loss.item())

    for epoch in range(sup_epochs):  # phase 2
        for batch in loader:
            with torch.no_grad():
                h = embedder(batch.to(device))
            loss_s = mse(supervisor(h[:, :-1, :]), h[:, 1:, :])
            opt_sup.zero_grad(); loss_s.backward(); opt_sup.step()
        log.info("SUP %d/%d L_S %.6f", epoch + 1, sup_epochs, loss_s.item())

    for epoch in range(gan_epochs):  # phase 3
        for batch in loader:
            batch = batch.to(device)
            noise = torch.randn(batch.size(0), t, latent_dim, device=device)
            h_real = embedder(batch).detach()
            h_fake = supervisor(generator(noise))
            d_fake = discriminator(h_fake)
            loss_u = bce(d_fake, torch.ones_like(d_fake))
            loss_s = mse(supervisor(h_real[:, :-1, :]), h_real[:, 1:, :])
            x_fake = recovery(h_fake)
            loss_v = (torch.mean(torch.abs(x_fake.mean(0) - batch.mean(0)))
                      + torch.mean(torch.abs(x_fake.std(0) - batch.std(0)))
                      + torch.mean(torch.abs(x_fake.std(1).mean(0) - batch.std(1).mean(0))))
            loss_g = loss_u + gamma * torch.sqrt(loss_s) + eta * loss_v
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            h = embedder(batch)
            loss_r = mse(recovery(h), batch)
            loss_s2 = mse(supervisor(h[:, :-1, :].detach()), h[:, 1:, :].detach())
            loss_e = 10.0 * torch.sqrt(loss_r) + 0.1 * loss_s2
            opt_e.zero_grad(); loss_e.backward(); opt_e.step()

            with torch.no_grad():
                h_real = embedder(batch)
                h_fake = supervisor(generator(noise))
            d_real, d_fake = discriminator(h_real), discriminator(h_fake)
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            if loss_d.item() > 0.15:  # do not over-train the discriminator
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
        log.info("GAN %d/%d D %.4f G %.4f L_S %.6f L_V %.4f", epoch + 1, gan_epochs,
                 loss_d.item(), loss_g.item(), loss_s.item(), loss_v.item())

    generator.eval(); supervisor.eval(); recovery.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, num_synth, 1000):
            b = min(1000, num_synth - start)
            noise = torch.randn(b, t, latent_dim, device=device)
            chunks.append(recovery(supervisor(generator(noise))).cpu().numpy())
    synthetic = np.concatenate(chunks).astype(np.float32)
    synthetic[:, :, MGEN_SLICE] = 0.0  # constant in the real data

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "generated_windows_scaled.npy", synthetic)
    torch.save({k: m.state_dict() for k, m in (("embedder", embedder), ("recovery", recovery),
                                                ("generator", generator), ("supervisor", supervisor),
                                                ("discriminator", discriminator))},
               output_dir / "timegan_weights.pth")

    rng = np.random.RandomState(seed)
    ref = data[np.sort(rng.choice(n, min(REFERENCE_SAMPLE, n), replace=False))].astype(np.float32)
    print("Within-window std, synthetic / real (random real sample from all clusters):")
    for name, i in sorted(INJECTION_TARGETS.items(), key=lambda kv: kv[1]):
        r, s = ref[:, :, i].std(axis=1).mean(), synthetic[:, :, i].std(axis=1).mean()
        print(f"  f{i:<2} {name:17s} real {r:.4f}  synthetic {s:.4f}  ratio {s / r:.3f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="EXPERIMENTAL TimeGAN generator (not released).")
    p.add_argument("--windows", type=Path, required=True, help="windows_scaled.npz or .npy")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/timegan"))
    p.add_argument("--ae_epochs", type=int, default=10)
    p.add_argument("--sup_epochs", type=int, default=10)
    p.add_argument("--gan_epochs", type=int, default=50)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=10.0, help="weight of the moment-matching loss")
    p.add_argument("--num_synth", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, extracted = open_windows(args.windows, args.output_dir)
    npy = extracted or args.windows
    try:
        train_timegan(Path(npy), data, args.output_dir, ae_epochs=args.ae_epochs,
                      sup_epochs=args.sup_epochs, gan_epochs=args.gan_epochs, gamma=args.gamma,
                      eta=args.eta, num_synth=args.num_synth, seed=args.seed)
    finally:
        del data
        if extracted is not None:
            os.remove(extracted)


if __name__ == "__main__":
    main()
