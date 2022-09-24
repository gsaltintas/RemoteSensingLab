# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 02/2021


from glob import glob
import os
import numpy as np

ORIGIN_FOLDER = '../week3/outputs' # folder containing points_*.npz
OUTPUT_FOLDER = './xyz_files'

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)



def convert_to_xyz(file_path, output_folder=OUTPUT_FOLDER):
    coords = np.load(file_path)['arr_0']
    output_path = os.path.join(output_folder, file_path.split('/')[-1]).replace('.npz', '.xyz')
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w') as f:
        f.write('X,Y,Z\n')
        for i in range(coords.shape[1]):
            f.write(f'{coords[0,i]},{coords[1,i]},{coords[2,i]}\n')


file_list = sorted(glob(os.path.join(ORIGIN_FOLDER, 'points_*.npz')))

for file in file_list:
    print(f'Processing {file}...')
    convert_to_xyz(file)
print('Done.')
