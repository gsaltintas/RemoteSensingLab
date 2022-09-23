# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 02/2021

################################################################################
# Lab 1 - Week 2 - Rational polynomial coefficients and ground control points 
#
## Tasks:
# * Task 1: in rpc.py, write function to project points into image coordinates using RPCs
# * Task 2: visually define precise image coordinates of new GCPs
#     - Use data/gcp_sketches.pdf to know where to click
# * Task 3: perform RPC refinement using an affine transform 
#     - Without check points (use all 16 points)
#     - With check points (use 6 points, check results with the other 10 points)
# * Task 4: calculate and plot residuals
#     - Without check points
#     - With check points
################################################################################


import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import os

# For interactive widgets
# If you are having problems, check for other options here and try a few: https://matplotlib.org/3.3.3/api/matplotlib_configuration_api.html#matplotlib.use
# matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
# matplotlib.use('WebAgg')

import parsers
import utils
import measure_points
from rpc_solution import *

# Use this to delete previously saved clicks (or you can delete clicks.npz)
FORCE_NEW_CLICKS = False
clicks_fname = 'clicks.npz'



################################################################################
# Step 1
# Compute image coordinates of given GCPs using RPCs and compare with given true coordinates.
# Go into rpc.py to fill in the necessary code.
################################################################################


# Read RPCs from metadata
coeff1, O1, I1 = parsers.read_coefficients('data/po_163003_pan_0000000_rpc.txt')
coeff2, O2, I2 = parsers.read_coefficients('data/po_163003_pan_0010000_rpc.txt')

# Read UTM coordinates for GCPs
gcp, gcp_name, _ = parsers.read_gcps('data/gcp.txt')


# Read given image coordinates for some control points
given_gcp_coordinates_img1, given_gcp_coordinates_img2 = parsers.read_pixel_coordinates('data/pixel_coordinates.txt')


# Indices corresponding to 5, 15, 17, 20, 31, 37, 61, 64, 66, 74
given_gcp_indices = [5,8,10,13,17,20,32,35,37,44]

# Initialize variables
gcp_rpc_projections_img1 = np.zeros((3, len(given_gcp_indices)))
gcp_rpc_projections_img2 = np.zeros((3, len(given_gcp_indices)))
for i, gcp_index in enumerate(given_gcp_indices):

    ########################################
    # TODO: Fill in missing code below
    ########################################

    phi,lbd = parsers.utm2deg(??????????,'32 N') # Convert UTM to latitude / longitude
    h = ?????????? # Get point height from GCP, no conversion needed
    X = [phi, lbd, h, 1] # 3D point in homogeneous coordinates

    # Apply RPC function to obtain image coordinates of GCP
    gcp_rpc_projections_img1[:,i] = rpc(??????????) # point coordinates in image 1
    gcp_rpc_projections_img2[:,i] = rpc(??????????) # point coordinates in image 2


# Print residuals
print('Residuals (should be only a few pixels off):')
print(utils.pointwise_RMSE(given_gcp_coordinates_img1, gcp_rpc_projections_img1))


################################################################################
# Step 2
# Get precise image coordinates of GCP by clicking on points.
################################################################################

################################################################################
# To try again, set FORCE_NEW_CLICKS=True or delete 'clicks.npz'
################################################################################

# Load images, skip if clicks will be loaded
if FORCE_NEW_CLICKS or not os.path.exists(clicks_fname):
    img1 = io.imread('data/po_163003_pan_0000000.tif')
    img2 = io.imread('data/po_163003_pan_0010000.tif')


# Points corresponding to 4, 38, 40, 60, 63, 78
clicked_gcp_indices = [4, 21, 23, 31, 34, 47]
corresponding_to = [4, 38, 40, 60, 63, 78]

# Initialize variables
new_gcp_rpc_projections_img1 = np.zeros((3, len(clicked_gcp_indices)))
clicked_points_img1 = np.zeros((3, len(clicked_gcp_indices)))
new_gcp_rpc_projections_img2 = np.zeros((3, len(clicked_gcp_indices)))
clicked_points_img2 = np.zeros((3, len(clicked_gcp_indices)))


for i, gcp_index in enumerate(clicked_gcp_indices):

    ########################################
    # TODO: Fill in missing code below
    ########################################

    phi_cp,lambda_cp = parsers.utm2deg(??????????,'32 N') # Latitude / longitude coordinates
    h_cp = ?????????? # Height
    X_cp = [phi_cp, lambda_cp, h_cp, 1] # Point in homogeneous coordinates
    
    # Apply RPC to obtain expected position of GCP in images
    new_gcp_rpc_projections_img1[:,i] = rpc(??????????)
    new_gcp_rpc_projections_img2[:,i] = rpc(??????????)

    # Click on points to obtain precise location of GCPs in images
    if FORCE_NEW_CLICKS or not os.path.exists(clicks_fname):
        clicked_points_img1[:,i] = measure_points.click_point(
            img1, 
            new_gcp_rpc_projections_img1[:,i], 
            title_addendum=' {}'.format(corresponding_to[i])
            )
        clicked_points_img2[:,i] = measure_points.click_point(
            img2, 
            new_gcp_rpc_projections_img2[:,i], 
            title_addendum=' {}'.format(corresponding_to[i])
            )

# Save or load clicks
if FORCE_NEW_CLICKS or not os.path.exists(clicks_fname):
    np.savez(clicks_fname, clicked_points_img1, clicked_points_img2)
    print('Clicks saved to clicks.npz')
else:
    npzfile = np.load(clicks_fname)
    clicked_points_img1 = npzfile['arr_0']
    clicked_points_img2 = npzfile['arr_1']
    print('Clicks loaded from clicks.npz')



################################################################################
# Step 3
# Compute RPC refinements using an affine transforms
################################################################################





################################################################################
# Part 1 - Use all 16 points to compute affine transform

# Concatenate indices, GCP image coordinates and GCP projections with RPCs for each image
all_indices = ?????????? # look into np.concatenate
all_gcp_coordinates_img1 = ?????????? # look into np.concatenate
all_gcp_coordinates_img2 = ?????????? # look into np.concatenate
all_gcp_rpc_projections_img1 = ?????????? # look into np.concatenate
all_gcp_rpc_projections_img2 = ?????????? # look into np.concatenate

# Calculate affine transforms for images 1 and 2 using utils.get_affine_transform
Aff_all_img1, _, _ = utils.get_affine_transform(??????????)
Aff_all_img2, _, _ = utils.get_affine_transform(??????????)

print('Affine transform computed using all 16 points')



################################################################################
# Part 2 - Compute affine with 6 best points, use other 10 as check points

# Find best 6 points (lowest RMSE) to use as control points, use other 10 points as check points
all_RMSEs_img1 = utils.pointwise_RMSE(??????????)
indices_control_img1 = np.argsort(all_RMSEs_img1)[:6]
indices_check_img1 = np.argsort(all_RMSEs_img1)[6:]
all_RMSEs_img2 = utils.pointwise_RMSE(??????????)
indices_control_img2 = np.argsort(all_RMSEs_img2)[:6]
indices_check_img2 = np.argsort(all_RMSEs_img2)[6:]

# Split variables according to indices lists
control_gcp_coordinates_img1 = ??????????
control_gcp_coordinates_img2 = ??????????
control_gcp_rpc_projections_img1 = ??????????
control_gcp_rpc_projections_img2 = ??????????
check_gcp_coordinates_img1 = ??????????
check_gcp_coordinates_img2 = ??????????
check_gcp_rpc_projections_img1 = ??????????
check_gcp_rpc_projections_img2 = ??????????


# Calculate affine transform using control points
Aff_control_img1, _, _ = utils.get_affine_transform(??????????)
Aff_control_img2, _, _ = utils.get_affine_transform(??????????)

print('Affine transform computed using 6 points')




################################################################################
# Step 4 - Calculate and plot residuals
################################################################################


# Check/create folder for outputs
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')

# Scale for quiver plots, don't change this (without asking first at least)
scale = 20.0

# Calculate residuals affine transform that was calculated using all points
residuals_all_img1 = utils.get_affine_residuals(??????????)
residuals_all_img2 = utils.get_affine_residuals(??????????)


################################################################################
# Quiver plots of residuals using all GCPs

# Image 1
try:
    plt.figure(figsize=(5, 10), dpi= 80)

    # Fill in these variables with the appropriate values
    # Check https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
    # Arrow origins x coordinates
    X = ?????????? # X.size = [16]
    # Arrow origins y coordinates
    Y = ?????????? # Y.size = [16]
    # Arrow x dimension
    U = ?????????? # U.size = [16]
    # Arrow y dimension
    V = ?????????? # V.size = [16]


    plt.quiver(X, Y, U, V, scale=scale, label='All points')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Residuals - all points - img1')
    plt.savefig('outputs/01-residuals-all-img1.pdf')
    plt.waitforbuttonpress()
    plt.close()
except:
    pass

# Image 2
try:
    plt.figure(figsize=(5, 10), dpi= 80)

    # Fill in these variables with the appropriate values
    # Check https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
    # Arrow origins x coordinates
    X = ?????????? # X.size = [16]
    # Arrow origins y coordinates
    Y = ?????????? # Y.size = [16]
    # Arrow x dimension
    U = ?????????? # U.size = [16]
    # Arrow y dimension
    V = ?????????? # V.size = [16]



    plt.quiver(X, Y, U, V, scale=scale, label='All points')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Residuals - all points - img2')
    plt.savefig('outputs/02-residuals-all-img2.pdf')
    plt.waitforbuttonpress()
    plt.close()
except:
    pass





################################################################################
# Quiver plots of residuals using 6 GCPs


# Calculate residuals for control and check points
residuals_control_img1 = utils.get_affine_residuals(??????????)
residuals_control_img2 = utils.get_affine_residuals(??????????)
residuals_check_img1 = utils.get_affine_residuals(??????????)
residuals_check_img2 = utils.get_affine_residuals(??????????)



# Image 1
try:
    plt.figure(figsize=(5, 10), dpi= 80)


    # Fill in these variables with the appropriate values
    # Check https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
    # Arrow origins x coordinates
    X = ?????????? # X.size = [6]
    # Arrow origins y coordinates
    Y = ?????????? # Y.size = [6]
    # Arrow x dimension
    U = ?????????? # U.size = [6]
    # Arrow y dimension
    V = ?????????? # V.size = [6]



    plt.quiver(X, Y, U, V, scale=scale, color='b', label='Control points')

    # Fill in these variables with the appropriate values
    # Check https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
    # Arrow origins x coordinates
    X = ?????????? # X.size = [10]
    # Arrow origins y coordinates
    Y = ?????????? # Y.size = [10]
    # Arrow x dimension
    U = ?????????? # U.size = [10]
    # Arrow y dimension
    V = ?????????? # V.size = [10]
    


    plt.quiver(X, Y, U, V, scale=scale, color='r', label='Check points')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Residuals - control/check points - img1')
    plt.savefig('outputs/03-residuals-control-check-img1.pdf')
    plt.waitforbuttonpress()
    plt.close()
except:
    pass

# Image 2
try:
    plt.figure(figsize=(5, 10), dpi= 80)


    # Fill in these variables with the appropriate values
    # Check https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
    # Arrow origins x coordinates
    X = ?????????? # X.size = [6]
    # Arrow origins y coordinates
    Y = ?????????? # Y.size = [6]
    # Arrow x dimension
    U = ?????????? # U.size = [6]
    # Arrow y dimension
    V = ?????????? # V.size = [6]



    plt.quiver(X, Y, U, V, scale=scale, color='b', label='Control points')
    

    # Fill in these variables with the appropriate values
    # Check https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html
    # Arrow origins x coordinates
    X = ?????????? # X.size = [10]
    # Arrow origins y coordinates
    Y = ?????????? # Y.size = [10]
    # Arrow x dimension
    U = ?????????? # U.size = [10]
    # Arrow y dimension
    V = ?????????? # V.size = [10]



    plt.quiver(X, Y, U, V, scale=scale, color='r', label='Check points')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Residuals - all points - img2')
    plt.savefig('outputs/04-residuals-control-check-img2.pdf')
    plt.waitforbuttonpress()
    plt.close()

except:
    pass
