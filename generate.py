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


generator = WGAN_sg_model.build_generator()
