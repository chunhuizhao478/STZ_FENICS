import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc

stz_data = np.loadtxt("data/mid_data182.txt")
curve = np.loadtxt("curve/mid_curve.txt")

#stz_data = np.loadtxt("mid_data226.txt")
#curve = np.loadtxt("mid_curve.txt")

plt.figure(1)

curve_3 = np.loadtxt("mid_curve_3.txt")
plt.plot(curve_3[:,0],curve_3[:,1])

#curve_5 = np.loadtxt("mid_curve_5.txt")
#plt.plot(curve_5[:,0],curve_5[:,1])

plt.plot(curve[:,0],curve[:,1])
plt.title("shear stress vs shear strain curve")
plt.xlabel("strain")
plt.ylabel("stress")
plt.xlim((0,0.03))
plt.ylim((0,2))
plt.show()

plt.figure(2)



"""
fig, ax = plt.subplots()
numofptr = np.shape(stz_data)[0]

radius = 0.6
sizes = np.ones(numofptr) * radius

index0 = np.argwhere(stz_data[:,2] == 0)
index0_locs = stz_data[index0,[0,1]]
index0_n = np.shape(index0_locs)[0]
sizes_0  = np.ones(index0_n) * radius

index1 = np.argwhere(stz_data[:,2] == 1)
index1_locs = stz_data[index1,[0,1]]
index1_n = np.shape(index1_locs)[0]
sizes_1  = np.ones(index1_n) * radius

index2 = np.argwhere(stz_data[:,2] == 2)
index2_locs = stz_data[index2,[0,1]]
index2_n = np.shape(index2_locs)[0]
sizes_2  = np.ones(index2_n) * radius

index3 = np.argwhere(stz_data[:,2] == 3)
index3_locs = stz_data[index3,[0,1]]
index3_n = np.shape(index3_locs)[0]
sizes_3  = np.ones(index3_n) * radius

patch_0 = [plt.Circle(center,size) for center, size in zip(index0_locs, sizes_0)]
patch_1 = [plt.Circle(center,size) for center, size in zip(index1_locs, sizes_1)]
patch_2 = [plt.Circle(center,size) for center, size in zip(index2_locs, sizes_2)]
patch_3 = [plt.Circle(center,size) for center, size in zip(index3_locs, sizes_3)]

collection_0 = mc.PatchCollection(patch_0, facecolors = "k")
collection_1 = mc.PatchCollection(patch_1, facecolors = "r")
collection_2 = mc.PatchCollection(patch_2, facecolors = "m")
collection_3 = mc.PatchCollection(patch_3, facecolors = "b")

ax.add_collection(collection_0)
ax.add_collection(collection_1)
ax.add_collection(collection_2)
ax.add_collection(collection_3)

ax.margins(0.01)
ax.set_aspect('equal', adjustable='box')
plt.show()
"""
