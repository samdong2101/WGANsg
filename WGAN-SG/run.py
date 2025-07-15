from WGAN_sg import WGAN_sg_model
from WGAN_sg.WGAN_sg_model import build_generator,build_discriminator
from WGAN_sg.WGAN_sg_model import GANS
from WGAN_sg import featurize
from WGAN_sg import convert_to_poscar
import pickle
import tensorflow as tf
import time
from datetime import date
from datetime import datetime
import numpy as np 
import random
from pymatgen.core.periodic_table import Element
from scipy.stats import wasserstein_distance

with open("/blue/hennig/sam.dong/structure_energies.pkl", "rb") as f:
    structures = pickle.load(f)
pre_process = featurize.PreprocessData(structures,['Zr','Cu','Al'],20,1)
structs = pre_process.preprocess_data()
png = featurize.PNGrepresentation(structs,None)
pngs,png_dim1,png_dim2,divisor_list,factor_list = png.featurize()
generator = build_generator(png_dim1,png_dim2,input_dim = 64)
discriminator = build_discriminator(png_dim1,png_dim2)
g_opt = tf.keras.optimizers.RMSprop(learning_rate = 0.0002)
d_opt = tf.keras.optimizers.RMSprop(learning_rate = 0.000077)
g_loss = WGAN_sg_model.BinaryCrossentropy()
d_loss = WGAN_sg_model.BinaryCrossentropy()
gans = GANS(generator,discriminator,input_dim = 64)
gans.compile(g_opt,d_opt,g_loss,d_loss)
def pool_coords(arr):
    pooled_coords = np.hstack(arr)
    x_coords = pooled_coords[:,0]
    y_coords = pooled_coords[:,1]
    z_coords = pooled_coords[:,2]
    return x_coords,y_coords,z_coords
def rescale_images(images,divisor_list,factor_list):
    for i in range(len(np.array(images))):
        for j in range(len(divisor_list)):
            images[i][j] = images[i][j] * divisor_list[j]/factor_list[j]
    return images

def extract_dims(images, elem_list):
    # Get atomic numbers of elements
    atomic_numbers = [Element(el).Z for el in elem_list]
    threshold = min(atomic_numbers) / 4

    dims = []
    for image in images:
        first_line = structure[0]
        count = sum(1 for pixel in first_line if pixel > threshold)
        dims.append(count)

    return dims
def extract_coords(images, dims, num_atoms):
    count = 0
    coords = []
    for dim in dims:
        if dim == num_atoms:
            coord = images[count][1:4][0:i]
            coords.append(coord)
        else:
            pass
        count = count + 1
    return np.array(coords)

emds = []
elem_list = 
for i in range(50):
    print(f'epoch {i*50}')
    gen_image_coords = []
    real_image_coords = []
    num_atoms = 2
    hist = gans.fit(batch1,
                    epochs=50,
                    batch_size = batch_size)
    gen_images = generator(tf.random.normal((1000,64,1)), training=True)
    gen_images = gen_images.numpy()
    real_images = random.sample(pngs,1000)rescale_images(real_images,divisor_list,factor_list)
    generated_images = rescale_images(gen_images,divisor_list,factor_list)
    real_images = rescale_images(real_images,divisor_list,factor_list)
    elem_list = pre_process.elem_list
    gen_dims = extract_dims(generated_images,elem_list = elem_list)
    real_dims = extract_dims(real_images,elem_list = elem_list)
    gen_coords = extract_coords(generated_images,gen_dims,num_atoms)
    real_coords = extract_coords(generated_images,gen_dims,num_atoms)
    x_gen,y_gen,z_gen = pool_coords(gen_coords)
    x_real,y_real,z_real = pool_coords(real_coords)
    emd_x = wasserstein_distance(x_real,x_gen)
    emd_y = wasserstein_distance(y_real,y_gen)
    emd_z = wasserstein_distance(z_real,z_gen)
    try:
        emd_means = [np.mean(emd) for emd in emds]
        if np.mean([emd_x,emd_y,emd_z]) < np.sort(emd_means[0]):
            now = datetime.now()
            now = str(now).replace(' ','_')
            print(f'saving generator weights with tag {now}')
            generator.save_weights(f'/blue/hennig/sam.dong/wgan-sg/generator_weights/Zr_Cu_Al_min_emd_{now}.h5')
    except:
        print('first iteration, no available data')
    emds.append([emd_x,emd_y,emd_z])

gen_images = generator(tf.random.normal((1000,64,1)), training=True)
rescaled_images = rescale_images(gen_images,divisor_list,factor_list)
convert = convert_to_poscar.POSCAR(test_images,['Zr','Cu','Al'],'/home/sam.dong/WGAN_sg/POSCARS/')
converted_structures = convert.convert_to_poscars()
