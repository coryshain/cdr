import os
import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
from dtsr.config import Config
from dtsr.util import filter_models

def stitch(dir_paths, image_names, output_path):
    fontpath = matplotlib.rcParams['datapath'] + '/fonts/ttf/DejaVuSans.ttf'
    imgs = []
    for dir_path in dir_paths:
        matches = [x for x in os.listdir(dir_path) if x in image_names]
        for match in matches:
            im = Image.open(dir_path + '/' + match)
            if im.mode == 'RGBA':
                im = im.convert('RGB')
            im = ImageOps.expand(im, 300, fill='white')
            x = im.size[0]
            pt = int(x / 50)
            font = ImageFont.truetype(fontpath, pt)
            draw = ImageDraw.Draw(im)
            draw.text((pt,pt), dir_path.split('/')[-1], fill=(0,0,0,0), font=font)
            imgs.append(im)
    if len(imgs) > 0:
        if len(imgs) > 1:
            append_images = imgs[1:]
        else:
            append_images = []
        imgs[0].save(output_path, 'PDF', resolution=100, save_all=True, append_images=append_images)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Stitches plots from DTSR models into a single PDF, making it easy to page through estimated IRF.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of models for which to stitch plots. Regex permitted. If unspecified, stitches all DTSR models.')
    argparser.add_argument('-i', '--image_names', nargs='+', default=['irf_atomic_scaled.png'], help='Name(s) of image file(s) to search for in each output directory.')
    argparser.add_argument('-o', '--output_name', type=str, default='DTSR_plots_stitched.pdf', help='Name of output file.')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)

    models = filter_models(p.model_list, args.models, dtsr_only=True)

    paths = []
    for m in models:
        path = p.outdir + '/' + m
        if os.path.exists(path):
            paths.append(path)

    stitch(paths, args.image_names, p.outdir + '/' + args.output_name)
