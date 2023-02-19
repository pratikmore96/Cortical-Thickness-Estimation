# -*- coding: utf-8 -*-
"""
@author: Pratik More, Akshay Panchal
"""

# import all needed libraries.
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import filters, morphology
from scipy import spatial

# Load all the data which is needed(Subject1, Subject2, Thickness_Data_Of_Subject1)
t1_s1 = nib.load('C:/Users/Pratik More/Downloads/raw_t1_subject_01.nii.gz').get_data()
t1_s2 = nib.load('C:/Users/Pratik More/Downloads/raw_t1_subject_02.nii.gz').get_data()
thickness_s1 = nib.load('C:/Users/Pratik More/Downloads/thickness_map_subject_01.nii.gz').get_data()
Replicated_data = np.copy(t1_s2)
t1_s2[t1_s2 < 80] = 0 # get white matter using tolerance
whiteMatter = t1_s2
med_wm = filters.median(whiteMatter) # Apply median filter on white matter.
dil = morphology.binary_dilation(med_wm) # Apply binary dilation on white matter.
wm_boundary = dil.astype(float) - med_wm # Subtract white matter from dilated white matter.
wm_boundary = wm_boundary.ravel() # convert it to 1 dimension
wm_boundary[wm_boundary != 1] = 0 # get the WM boundary by replacing the coordinates which are not in wmboundary with zero.
wm_boundary = np.reshape(wm_boundary,(256,256,256)) # Reshape the wm boundary array in 3D
#-------------------------------------------
dil_new = np.copy(dil) # create a copy of dilated boundary.
for i in range(5):
    dil_new = morphology.binary_dilation(dil_new) # dilate wm few times to cover the whole area of grey matter.
edge_t2_dil = dil_new.astype(float) - med_wm # delete the original wm from the dilated one.
ravel_d2_edge  = edge_t2_dil.ravel() # convert it to 1 dimension
ravel_d2_edge[ravel_d2_edge != 1] = 0 # get the grey matter.
ravel_d2_edge = np.reshape(ravel_d2_edge,(256,256,256))  # reshape it to 3D.
#----------------------------------------
d1_greymatter_ravel = ravel_d2_edge.ravel() # convert it to 1 dimension
d1_org_ravel = Replicated_data.ravel() # convert it to 1 dimension
grey_arr_int_3d = np.zeros(d1_org_ravel.size)
index_of_1_greymatter = np.where(d1_greymatter_ravel == 1) # take those points on which we have to process.
# To find out original greymatter with original values.
for i in index_of_1_greymatter[0]:
    np.put(grey_arr_int_3d, i, d1_org_ravel[i])     
sum_intensity = np.sum(grey_arr_int_3d)
grey_arr_int_3d = np.reshape(grey_arr_int_3d,(256,256,256))
sum_intensity = []
for i in range(256):
    d2_edge = grey_arr_int_3d[:,i,:]
    grey_matter_intensity = sum(map(sum,d2_edge))
    sum_intensity.append(grey_matter_intensity)
#--------------------------------------------------
#apply tolerance to extract grey matter.
grey_arr_int_3d_replica = np.copy(grey_arr_int_3d)
grey_arr_int_3d_replica[grey_arr_int_3d_replica < 40] = 0
#-------------------------------------------------
#get the pile boundary.
pile_boundary = morphology.binary_dilation(grey_arr_int_3d_replica) # do binary dilation on grey matter.
pile_boundary = pile_boundary.astype(float) - grey_arr_int_3d_replica # subtract original from dilated.
pile_boundary  = pile_boundary.ravel() # Convert to 1D
pile_boundary[pile_boundary != 1] = 0 # get the pile boundary and white matter boundary.
pile_boundary = np.reshape(pile_boundary,(256,256,256)) # convert to 3D
#-------------------------------------------------
# get only pile boundary and subtract white matter boundary
pile_boundary_replica = pile_boundary.astype(float) - dil.astype(float)
pile_boundary_replica  = pile_boundary_replica.ravel()  # Convert to 1D
pile_boundary_replica[pile_boundary_replica != 1] = 0 # get the pile boundary
pile_boundary_replica = np.reshape(pile_boundary_replica,(256,256,256)) # convert to 3D
#---------------------------------------------------
# creating array for with dimensions for wm boundary.
wm_boundary_final = np.copy(wm_boundary[:,:,:])
pile_boundary_replica_final = np.copy(pile_boundary_replica[:,:,:])

pile_x, pile_y, pile_z = np.where(pile_boundary_replica_final == 1)
wm_x, wm_y, wm_z = np.where(wm_boundary_final == 1)

# create array of size N*3 (N=no of elements, 3 is x,y,z axis), which have all the coordinates of Pile and white matter.
pile_coordinate = []
wm_coordinate = []

for x in range(pile_x.size):
    arr = [pile_x[x], pile_y[x], pile_z[x]]
    pile_coordinate.append(arr)
    
pile_coordinate = np.array(pile_coordinate,ndmin=2)

for x in range(wm_x.size):
    arr = [wm_x[x], wm_y[x], wm_z[x]]
    wm_coordinate.append(arr)
    
wm_coordinate = np.array(wm_coordinate,ndmin=2)

# find min distance along with nearest point using KD distanceTree
distanceTree = spatial.KDTree(pile_coordinate)
minimumDist, minimumId = distanceTree.query(wm_coordinate)

#replacing WM boundary co-ordinates with the minimum distance from pile boundary cooridnates.
l = 0
for wm in wm_coordinate: 
    wm = wm.tolist()
    wm_boundary_final[wm[0],wm[1],wm[2]] = minimumDist[l]
    l+=1
#-------------------------------------------------
# creating array for grey matter.
grey_scale = np.copy(grey_arr_int_3d_replica[:,:,:])
grey_x, grey_y, grey_z = np.where(grey_scale != 0)

# create array of size N*3 (N=no of elements, 3 is x,y,z axis), which have all the coordinates of grey matter.
grey_coordinate = []

for x in range(grey_x.size):
    arr = [grey_x[x], grey_y[x], grey_z[x]]
    grey_coordinate.append(arr)
    
grey_coordinate = np.array(grey_coordinate,ndmin=2)

# find min distance along with nearest point using KD distanceTree
distanceTree = spatial.KDTree(wm_coordinate)
minimumDist, minimumId = distanceTree.query(grey_coordinate)

# Replace the grey matter coordiante values with the smallest distance with white matter coordinate value
l = 0
for grey in grey_coordinate: 
    grey = grey.tolist()
    grey_scale[grey[0],grey[1],grey[2]] = wm_boundary_final[wm_coordinate[minimumId[l]][0],wm_coordinate[minimumId[l]][1],wm_coordinate[minimumId[l]][2]]
    l+=1

# Create a copy to final grey matter to apply filters and remove the excess area
thicknessMapSubject2 = np.copy(grey_scale) # create copy
thicknessMapSubject2[thicknessMapSubject2 > 6] = 0 # apply tolerance
thicknessMapSubject2 = filters.gaussian(thicknessMapSubject2, sigma=0.3) # apply filter

# Print final result
plt.imshow(thicknessMapSubject2[:,80,:])

