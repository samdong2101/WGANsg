# WGANs-Integrated-Genetic-Algorithms
This repository contains the code of the Wasserstein Generative Adversarial Structure Generator (WGAN-sg) designed to create stronger starting initial populations for genetic algorithms for crystal structure and phase prediction. 

<img width="2500" height="1045" alt="generative_initialization_strategy_12_2_2024_d" src="https://github.com/user-attachments/assets/1d6132da-4b8b-4d8d-a7fa-fbcccf3dc25a" />

The WGAN-sg is trained on thermodynamically stable and metastable crystals (<0.1 eV/atom), where crystals are represented in a png image format. This png representation takes crystal parameters such as element type, fractional coordinates, lattice parameters etc. and encodes these values into the pixels of an image, where pixel brightness indicates magnitude. 

<img width="1555" height="1224" alt="descriptor_with_training_loop2" src="https://github.com/user-attachments/assets/e129f64e-61b2-466b-a61b-242250d77ca0" />

- Data preparation and model training can be executed within run.py, which can be executed via 
```python run.py --config /path/to/input_file``` 
- The input file for run.py is a json containing a dictionary of parameters, an example of the input file can be found in this repository under 
```WGANsg/input.json```
- The contents within the input file are as follows
    - structures_path: a path to an array-type of pymatgen structures
    - elem_list: A list of strings containing atomic symbols of your desired composition space i.e. ['Cu','Al']
    - max_atoms: the maximum number of atoms N you want in your data
    - min_atoms: the minimum number of atoms N you want in your data
    - g_lr: the learning rate of the generator network
    - c_lr: the learning rate of the critic network
    - epochs_inner: the number of epochs you would like for training before each evaluation loop
    - epochs_outer: the total number of evaluation loops you would like for training
    - batch_size: batch size of data
    - generator_weights_path: the desired destination to save the weights of your generator network
    - num_images: the total number of images (crystal structures) you want to generate
    - poscar_path: the desired destination of your converted crystal structures made by the WGAN-sg 
