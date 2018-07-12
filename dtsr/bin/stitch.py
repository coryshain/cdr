import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
from dtsr.config import Config

def stitch(dir_paths, filename, output_path):
     imgs = []
     for dir_path in dir_paths:
         im = Image.open(dir_path + '/' + filename)
         if im.mode == 'RGBA':
             im = im.convert('RGB')
         im = ImageOps.expand(im, 300, fill='white')
         x = im.size[0]
         pt = int(x / 50)
         font = ImageFont.truetype("arial.ttf", pt)
         draw = ImageDraw.Draw(im)
         draw.text((pt,pt), dir_path.split('/')[-1], fill=(0,0,0,0), font=font)
         imgs.append(im)
     imgs[0].save(output_path, 'PDF', resolution=100, save_all=True, append_images=imgs[1:])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
            Stitches plots from DTSR models into a single PDF, making it easy to page through estimated IRF.
        ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-i', '--irftype', type=str, default='atomic_scaled', help='Type of plot to paste. Choose from "atomic_scaled", "atomic_unscaled", "composite_scaled", and "composite_unscaled".')
    argparser.add_argument('-M', '--mc', action='store_true', help='Use Monte Carlo (MC) IRF plots with credible intervals (DTSRBayes only).')
    args, unknown = argparser.parse_known_args()

    assert args.irftype in ["atomic_scaled", "atomic_unscaled", "composite_scaled", "composite_unscaled"], 'Unrecognized argument "%s" to -i/--irftype' %args.irftype

    p = Config(args.config_path)
    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    paths = []
    for m in models:
        path = p.outdir + '/' + m
        if os.path.exists(path):
            paths.append(path)

    if args.mc:
        stitch(paths, 'mc_irf_' + args.irftype + '.png', p.outdir + '/irf_' + args.irftype + '.pdf')
    else:
        stitch(paths, 'irf_' + args.irftype + '.png', p.outdir + '/irf_' + args.irftype + '.pdf')
