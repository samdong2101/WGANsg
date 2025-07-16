from WGAN_sg import WGAN_sg_model
from featurize import struct2img
from post_process import img2struct
import pickle
import tensorflow as tf
import time
from datetime import date
from datetime import datetime
import numpy as np
import random
from pymatgen.core.periodic_table import Element
from scipy.stats import wasserstein_distance
import argparse
import json
import time


def main():
  parser = argparse.ArgumentParser(description="Generating materials with JSON config.")
  parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to JSON config file, e.g., config.json'
    )
  args = parser.parse_args()
  with open(args.config) as f:
      config = json.load(f)
  structures_path = config.get("structures_path")
  elem_list = config.get("elem_list")
  max_atoms = config.get("max_atoms")
  min_atoms = config.get("min_atoms")
  pretrained_path = config.get("pretrained_path")
  num_images = config.get("num_images")
  poscar_path = config.get("poscar_path")

  with open(structures_path, "rb") as f:
    structures = pickle.load(f)
  pre_process = struct2img.PreprocessData(structures,elem_list,max_atoms,min_atoms)
  structs = pre_process.preprocess_data()
  png = struct2img.PNGrepresentation(structs,None)
  pngs,png_dim1,png_dim2,divisor_list,factor_list = png.featurize()

  generator = WGAN_sg_model.build_generator(png_dim1,png_dim2,64)
  generator = generator.load_weights(pretrained_path)
  then = time.time()
  gen_images = generator(tf.random.normal((num_images,64,1)), training=True)
  rescaled_images = rescale_images(gen_images,divisor_list,factor_list)
  convert = img2struct.POSCAR(rescaled_images,elem_list,poscar_path)
  converted_structures = convert.convert_to_poscars()
  now = time.time()
  diff = now - then 
  print(f"successfully generated and converted {num_images} structures located at {poscar_path} in {diff} seconds!")

if __name__ == '__main__':
    main()

