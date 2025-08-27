"""
Infer HRNet config from a checkpoint and try to instantiate/load a compatible PoseHighResolutionNet.
Saves a best-effort model_state_dict to checkpoints/compat_hrnet_from_ckpt.pth and prints load diagnostics.
"""
import torch
from pathlib import Path
import re
import json

ckpt_path = Path('hrnet_w18_small_model_v1.pth')
if not ckpt_path.exists():
    print('Checkpoint not found:', ckpt_path)
    raise SystemExit

ck = torch.load(str(ckpt_path), map_location='cpu')
if isinstance(ck, dict) and 'state_dict' in ck:
    sd = ck['state_dict']
elif isinstance(ck, dict) and 'model' in ck:
    sd = ck['model']
else:
    sd = ck

# Remove module. prefix
new_sd = {}
for k, v in sd.items():
    nk = k[7:] if k.startswith('module.') else k
    new_sd[nk] = v
sd = new_sd

# helper to find shapes
def find_weight(name_pattern):
    for k, v in sd.items():
        if re.search(name_pattern, k):
            return k, v.shape
    return None, None

# Infer branch channel counts for stage2/3/4
inferred = {}
for stage in [2,3,4]:
    stage_key = f'stage{stage}.0.branches.'
    # collect branch indices by scanning keys
    branches = {}
    for k in sd.keys():
        m = re.match(rf'stage{stage}\.0\.branches\.(\d+)\.0\.conv1\.weight', k)
        if m:
            b = int(m.group(1))
            branches[b] = sd[k].shape[0]  # out channels
    if branches:
        # make a list ordered by branch index
        inferred[f'stage{stage}'] = [branches[i] for i in sorted(branches.keys())]
    else:
        inferred[f'stage{stage}'] = None

print('Inferred branch channels:')
print(json.dumps(inferred, indent=2))

# Build config using inferred channels; fall back to hrnet_w18_small defaults if missing
from hrnet_model import hrnet_w18_small_config, get_pose_net
cfg = dict(hrnet_w18_small_config)  # shallow copy
cfg['NUM_JOINTS'] = 14
cfg['PRETRAINED_LAYERS'] = ['*']

if inferred['stage2']:
    cfg['STAGE2'] = dict(cfg['STAGE2'])
    cfg['STAGE2']['NUM_CHANNELS'] = inferred['stage2']
if inferred['stage3']:
    cfg['STAGE3'] = dict(cfg['STAGE3'])
    cfg['STAGE3']['NUM_CHANNELS'] = inferred['stage3']
if inferred['stage4']:
    cfg['STAGE4'] = dict(cfg['STAGE4'])
    cfg['STAGE4']['NUM_CHANNELS'] = inferred['stage4']

print('\nUsing config:')
print(json.dumps({'STAGE2':cfg['STAGE2']['NUM_CHANNELS'], 'STAGE3':cfg['STAGE3']['NUM_CHANNELS'], 'STAGE4':cfg['STAGE4']['NUM_CHANNELS']}, indent=2))

# Instantiate model
model = get_pose_net(cfg, is_train=False)
model_sd = model.state_dict()
print('\nModel param count:', sum(p.numel() for p in model.parameters()))

# Try to match keys by intersection and strict load diagnostics
ck_keys = set(sd.keys())
model_keys = set(model_sd.keys())
common = ck_keys & model_keys
print(f'Common keys: {len(common)} / {len(model_keys)} model keys')

# Prepare new state dict to load
load_sd = {}
adapted = 0
skipped = 0
for k in model_keys:
    if k in sd and sd[k].shape == model_sd[k].shape:
        load_sd[k] = sd[k]
        adapted += 1
    else:
        skipped += 1

print(f'Will load {adapted} exact-shape params, skip {skipped}')
# Load non-strict
model_sd.update(load_sd)
model.load_state_dict(model_sd, strict=False)

out_path = Path('checkpoints')
out_path.mkdir(parents=True, exist_ok=True)
torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, out_path / 'compat_hrnet_from_ckpt.pth')
print('Saved compat model to', out_path / 'compat_hrnet_from_ckpt.pth')

# Quick forward check
import torch
dummy = torch.randn(1,3,256,256)
try:
    y = model(dummy)
    print('Forward output shape:', y.shape)
except Exception as e:
    print('Forward failed:', e)

print('Done')
