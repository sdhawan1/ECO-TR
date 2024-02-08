import os
import sys
import cv2
import matplotlib.pyplot as plt
from dotmap import DotMap
SCRIPT_DIR = sys.path.append(
    os.path.join((os.path.dirname(os.path.abspath(__file__))),'..'))
os.sys.path.append(SCRIPT_DIR)
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.models.ecotr_engines import *
from src.models.utils.utils import *

import argparse
import time
from pathlib import Path

if __name__ == '__main__':
    # handle input arguments:
    parser = argparse.ArgumentParser()
    # input assumption 1: put the ref's in one folder & call them "<prefix>_reference.xyz"
    parser.add_argument('reference_folder_name', type=str, help="path to reference images")
    # input assumption 2: put the ref's in one folder & call them "<prefix>_target.xyz"
    parser.add_argument('target_folder_name', type=str, help="path to target images")
    parser.add_argument('res_height', type=int, default=500, help="input image effective height")
    parser.add_argument('aspect_ratio', type-float, default=1.5, help="width/height ratio to use at inference")
    parser.add_argument('n_keypoints', type=int, default=1000, help="# keypoints to correlate between images")
    parser.add_argument('output_path', type=str, default='.', help="where to save outputs.")
    args = parser.parse_args()

    # save the timings in a cumulative list:
    inference_runtimes = []

    # TODO: automatically parse target & reference img paths and connect them.
    p_reference = Path(args.reference_folder_name)
    for ref_img in p_reference.glob('*.*'):
        img0=cv2.imread(ref_img)
        prefix = ref_img.name.split('_reference')[0]

        # find corresponding target image.
        tgts = list(Path(args.target_folder_name).glob(f'{prefix}_target*'))
        if len(tgts) == 0:
            print(f"image: {prefix}_target not found!")
            break
        else:
            print(f"starting to parse {prefix}_target!")
            print(tgts[0])
        img1=cv2.imread(tgts[0])
        
        dict = DotMap(lower_config(get_cfg_defaults()))
        fix_randomness(42)
        engine=ECOTR_Engine(dict.ecotr)
        engine.use_cuda=True
        engine.load_weight('cuda')

        engine.MAX_KPTS_NUM=args.n_keypoints
        engine.ASPECT_RATIOS=[1.0,args.aspect_ratio]  # keeping this constant for now, we can change it later?

        # output 1: write the output timings somewhere!
        start = time.time()
        matches=engine.forward(img0,img1,cycle=False,level='fine')
        # xx=engine.forward_2stage(img0,img1,cycle=False)
        end = time.time()
        inference_runtimes.append(end-start)

        matches=matches[matches[:,-1]<1e-2]
        canvas = draw_matches(img0,img1,matches.copy(),'dot')
        height,width = canvas.shape[:2]

        # TODO: create a different name for every output file corresponding to 'prefix'.
        filename = f'{prefix}_output.jpg'
        output_file = Path(args.output_path) / filename
        cv2.imwrite(output_file,canvas)

        #dpi=100
        #figsize = width / float(dpi), height / float(dpi)
        #plt.figure(figsize=figsize)
        #plt.imshow(canvas)
        #plt.show()

        # output 2: print out the average runtime!
