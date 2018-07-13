import os
import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
from dtsr.config import Config

def stitch(dir_paths, filename_suffix, output_path):
    fontpath = matplotlib.rcParams['datapath'] + '/fonts/ttf/DejaVuSans.ttf'
    imgs = []
    for dir_path in dir_paths:
        matches = [x for x in os.listdir(dir_path) if x.endswith(filename_suffix)]
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
    imgs[0].save(output_path, 'PDF', resolution=100, save_all=True, append_images=imgs[1:])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
            Stitches plots from DTSR models into a single PDF, making it easy to page through estimated IRF.
        ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-i', '--image_suffix', type=str, default='irf_atomic_scaled.png', help='Name of image file to search for in each output directory.')
    args, unknown = argparser.parse_known_args()

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

    stitch(paths, args.image_suffix, p.outdir + '/' + '.'.join(args.image_suffix.split('.')[:-1]) + '.pdf')
