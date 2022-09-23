# Parsers for IKONOS data adapted from matlab code

import numpy as np
import os

def read_coefficients(filename : str):
    assert(os.path.exists(filename))
    file = open(filename, 'r')
    val = np.zeros(10)
    for i in range(10):
        line = file.readline()
        val[i] = float(line.split(' ')[1])
    coeff = np.zeros((20,4))
    for i in range(4):
        for j in range(20):
            line = file.readline()
#             print(line.split(' ')[1])
            coeff[j,i] = float(line.split(' ')[1])
    O = np.zeros((4,4))
    O[0,0] = val[7]
    O[1,1] = val[8]
    O[2,2] = val[9]
    O[0,3] = val[2]
    O[1,3] = val[3]
    O[2,3] = val[4]
    O[3,3] = 1
    I = np.zeros((3,3))
    I[0,0] = val[6]
    I[1,1] = val[5]
    I[0,2] = val[1]
    I[1,2] = val[0]
    I[2,2] = 1
    return coeff, O, I

def read_gcps(filename : str):    
    assert(os.path.exists(filename))
    file = open(filename, 'r')
    name = []
    gcp = np.zeros((3,0))
    what_is_this = np.zeros((3,0))
    for i in range(1000):
        line = file.readline()
        if not line:
            break 
        line = line.replace('\n', '').split('\t')
        new_gcp = np.expand_dims(np.array([float(line[1]), float(line[2]), float(line[3])]), 1)
        gcp = np.concatenate((gcp, new_gcp), axis=1)
        name.append(line[0])
        new_wit = np.expand_dims(np.array([float(line[4]), float(line[5]), float(line[6])]), 1)
        what_is_this = np.concatenate((what_is_this, new_wit), axis=1)
    return gcp, name, what_is_this

def read_pixel_coordinates(filename : str):
    assert(os.path.exists(filename))
    file = open(filename, 'r')
    coordinates = np.zeros((2,0))
    for i in range(1000):
        line = file.readline()
        if not line:
            break 
        line = line.replace('\n', '').split('\t')
        coordinates = np.concatenate((coordinates, np.expand_dims(np.array([float(line[2]), float(line[3])]), 1)), axis=1)
    
    coordinates = euc2hom(coordinates)
    
    im1_coordinates = coordinates[:,0::2]
    im2_coordinates = coordinates[:,1::2]

    return im1_coordinates, im2_coordinates

def euc2hom(coordinates):
    return np.concatenate((coordinates, np.ones((1,coordinates.shape[1]))), axis=0)

def utm2deg(x, y, utmzone):
    if utmzone[-1] > 'M':
        hemisphere = 'N'
    else:
        hemisphere = 'S'
    zone = float(utmzone[:2])
    sa = 6378137.0
    sb = 6356752.314245
    e2 = (((sa ** 2 ) - ( sb ** 2 )) ** 0.5) / sb
    e2sq = e2 ** 2
    c = (sa ** 2 ) / sb
    xx = x - 500000
    if hemisphere == 'S':
        yy = y - 10000000
    else:
        yy = y
    ss = ((zone * 6) - 183)
    lat =  yy / (6366197.724 * 0.9996)
    v = (c / ((1 + (e2sq * (np.cos(lat)) ** 2))) ** 0.5) * 0.9996
    a = xx / v
    a1 = np.sin(2 * lat)
    a2 = a1 * (np.cos(lat)) ** 2
    j2 = lat + (a1 / 2)
    j4 = ((3 * j2) + a2) / 4
    j6 = ((5 * j4) + (a2 * (np.cos(lat)) ** 2)) / 3
    alpha = (3 / 4) * e2sq
    beta = (5 / 3) * alpha ** 2
    gamma = (35 / 27) * alpha ** 3
    Bm = 0.9996 * c * (lat - alpha * j2 + beta * j4 - gamma * j6)
    b = (yy - Bm) / v
    Epsi = ((e2sq * a ** 2 ) / 2 ) * (np.cos(lat)) ** 2
    Eps = a * (1 - (Epsi / 3))
    nab = (b * (1 - Epsi)) + lat
    senoheps = (np.exp(Eps) - np.exp(-Eps) ) / 2
    Delt = np.arctan(senoheps / (np.cos(nab)))
    TaO = np.arctan(np.cos(Delt) * np.tan(nab))
    lon = (Delt * (180 / np.pi)) + ss
    lat = (lat + (1 + e2sq * (np.cos(lat) ** 2) - (3 / 2) * e2sq * np.sin(lat) * np.cos(lat) * (TaO - lat)) * (TaO - lat)) * (180 / np.pi)
    return lat, lon