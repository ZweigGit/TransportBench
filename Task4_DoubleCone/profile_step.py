"""Time one golden-protocol training step of the redesigned wien_deeponet."""
import time
import torch
from data_loader import get_dataloader_and_stats
from train import build_model

dev = torch.device('cuda')
train_loader, _, x_norm, y_norm, vac = get_dataloader_and_stats(
    '../data/double_cone_dataset.pt', 9, dev)
ctx = {'n_param': 4, 'vac': vac.to(dev)}
m = build_model('wien_deeponet', False, ctx).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

x, y = next(iter(train_loader))
xe, ye = x_norm.encode(x), y_norm.encode(y)

def sync():
    torch.cuda.synchronize()

# warmup
for _ in range(3):
    opt.zero_grad(); m(xe, ye); loss = m.aux_loss; loss.backward(); opt.step()
sync()

t = {}
for tag, fn in [
    ('forward(x,y) incl dwt loss', lambda: m(xe, ye)),
    ('aux read', lambda: m.aux_loss),
]:
    sync(); t0 = time.perf_counter()
    for _ in range(10):
        out = fn()
    sync(); t[tag] = (time.perf_counter() - t0) / 10

sync(); t0 = time.perf_counter()
for _ in range(10):
    opt.zero_grad(); m(xe, ye); loss = m.aux_loss; loss.backward()
sync(); t['fwd+bwd'] = (time.perf_counter() - t0) / 10

t0 = time.perf_counter()
for _ in range(10):
    opt.step()
sync(); t['opt.step'] = (time.perf_counter() - t0) / 10

# forward WITHOUT y (eval path)
m.eval()
sync(); t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        m(xe)
sync(); t['forward(x) eval'] = (time.perf_counter() - t0) / 10

# isolate the loss internals: dwt2_stack + support_weights on y
from wavelet_haar import dwt2_stack, support_weights
hw = [(17, 384), (9, 192), (5, 96), (3, 48), (2, 24)]
sync(); t0 = time.perf_counter()
for _ in range(10):
    A_t, Ds_t, _ = dwt2_stack(ye, 4, 'haar')
    w = support_weights(~ctx['vac'], hw)
sync(); t['dwt2_stack(y)'] = (time.perf_counter() - t0) / 10

for k, v in t.items():
    print(f'{k:32s} {v*1000:9.1f} ms')
print(f'{"full step estimate":32s} {(t["fwd+bwd"]+t["opt.step"])*1000:9.1f} ms')
