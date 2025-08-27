from model_utils import load_model_safely
import torch

model = load_model_safely('checkpoints/compat_hrnet_from_ckpt.pth', target_keypoints=14, device='cpu')
for i, (n, p) in enumerate(model.named_parameters()):
    print(i, n, p.shape, 'requires_grad=', p.requires_grad)

print('\nSample parameter name prefixes:')
seen = set()
for n, p in model.named_parameters():
    pref = n.split('.')[0]
    seen.add(pref)
print(sorted(list(seen))[:50])
