"""
Inspect saved summed heatmap overlay images and report simple statistics.
Usage:
  python inspect_heatmaps.py --dir outputs/diagnostics/heatmaps --max 6
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def analyze_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    mx = int(gray.max())
    frac_bright = float((gray > 10).sum()) / (gray.size)
    # concentration: fraction of mass in top 5% pixels
    thresh = np.percentile(gray, 95)
    top_frac = float((gray >= thresh).sum()) / (gray.size)
    return {'path': str(path), 'mean': mean, 'max': mx, 'frac_bright': frac_bright, 'top_frac': top_frac}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='outputs/diagnostics/heatmaps')
    parser.add_argument('--max', type=int, default=6)
    args = parser.parse_args()

    d = Path(args.dir)
    if not d.exists():
        print(f"Directory not found: {d}")
        return

    files = sorted(list(d.glob('*_summed.png')))
    if not files:
        print('No summed heatmap images found in', d)
        return

    files = files[:args.max]
    results = []
    for f in files:
        r = analyze_image(f)
        if r:
            results.append(r)
    # Print concise table
    for r in results:
        print(f"{Path(r['path']).name}: mean={r['mean']:.1f}, max={r['max']}, frac_bright={r['frac_bright']*100:.2f}%, top5%_frac={r['top_frac']*100:.2f}%")

    # Summary heuristics
    concentrated = [r for r in results if r['top_frac'] < 0.01 and r['max'] > 30]
    diffuse = [r for r in results if r['top_frac'] >= 0.01 or r['max'] <= 30]

    print('\nSummary:')
    print(f"  Sampled {len(results)} images")
    print(f"  Concentrated heatmaps (small bright region & high max): {len(concentrated)}")
    print(f"  Diffuse/weak heatmaps: {len(diffuse)}")

if __name__ == '__main__':
    main()
