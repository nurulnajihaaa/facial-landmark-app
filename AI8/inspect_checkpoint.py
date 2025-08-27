import torch
from pathlib import Path
import sys

p = Path('hrnet_w18_small_model_v1.pth')
if not p.exists():
    print('Checkpoint not found:', p)
    sys.exit(1)

ck = torch.load(str(p), map_location='cpu')
# get inner dict
if isinstance(ck, dict) and ('state_dict' in ck or 'model' in ck):
    if 'state_dict' in ck:
        sd = ck['state_dict']
    else:
        sd = ck.get('model', ck)
else:
    sd = ck

print('Total keys:', len(sd))
# Print first 80 keys and shapes
for i, (k, v) in enumerate(sd.items()):
    if hasattr(v, 'shape'):
        print(i, k, tuple(v.shape))
    else:
        print(i, k, type(v))
    if i >= 80:
        break

# Print sample of conv layers and transition layers
print('\nSample keys containing "layer1" or "transition" or "conv" or "final"')
for k in sd.keys():
    if any(s in k for s in ['layer1', 'transition', 'final', 'conv', 'bn']):
        if hasattr(sd[k], 'shape'):
            print(k, tuple(sd[k].shape))
        else:
            print(k, type(sd[k]))

print('\nDone')
