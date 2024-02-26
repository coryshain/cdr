import os
import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
from cdr.config import Config
from cdr.util import filter_models, filter_names

def stitch(dir_paths, image_names, output_path):
    fontpath = matplotlib.rcParams['datapath'] + '/fonts/ttf/DejaVuSans.ttf'
    imgs = []
    for dir_path in dir_paths:
        matches = filter_names(os.listdir(dir_path), image_names)
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
            if os.path.exists(output_path):
                append = True
            else:
                append = False
            im.save(output_path, 'PDF', resolution=100, save_all=True, append=append)
            imgs.append(im)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Stitches plots from CDR models into a single PDF, making it easy to page through estimated IRFs.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of models for which to stitch plots. Regex permitted. If unspecified, stitches all CDR models.')
    argparser.add_argument('-i', '--image_names', nargs='+', default=['irf_atomic_scaled.png'], help='Name(s) of image file(s) to search for in each output directory. Regex matching supported.')
    argparser.add_argument('-o', '--output_name', type=str, default='CDR_plots_stitched.pdf', help='Name of output file.')
    args = argparser.parse_args()

    p = Config(args.config_path)

    models = filter_models(p.model_names, args.models, cdr_only=True)

    paths = []
    for m in models:
        path = p.outdir + '/' + m.replace(':', '+')
        if os.path.exists(path):
            paths.append(path)

    if os.path.exists(args.output_name):
        os.remove(args.output_name)
    stitch(paths, args.image_names, args.output_name)
