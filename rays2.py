from matplotlib import pyplot
from mpl_toolkits import mplot3d
figure = pyplot.figure(figsize=(15,15))
axes = mplot3d.Axes3D(figure)
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X
def plotCubeAt2(positions,sizes=None, **kwargs):
    if not isinstance(sizes,(list,np.ndarray)):
        sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s in zip(positions,sizes):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g), **kwargs)


def ray_intersect_triangle(p0, p1, triangle):
    # Tests if a ray starting at point p0, in the direction
    # p1 - p0, will intersect with the triangle.
    #
    # arguments:
    # p0, p1: numpy.ndarray, both with shape (3,) for x, y, z.
    # triangle: numpy.ndarray, shaped (3,3), with each row
    #     representing a vertex and three columns for x, y, z.
    #
    # returns: 
    #    0.0 if ray does not intersect triangle, 
    #    1.0 if it will intersect the triangle,
    #    2.0 if starting point lies in the triangle.

    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)

    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)

    # Here is the main difference with the code in the link.
    # Instead of returning if the ray is in the plane of the 
    # triangle, we set rI, the parameter at which the ray 
    # intersects the plane of the triangle, to zero so that 
    # we can later check if the starting point of the ray
    # lies on the triangle. This is important for checking 
    # if a point is inside a polygon or not.
    
    if (b == 0.0):
        # ray is parallel to the plane
        if a != 0.0:
            # ray is outside but parallel to the plane
            return 0
        else:
            # ray is parallel and lies in the plane
            rI = 0.0
    else:
        rI = a / b

    if rI < 0.0:
        return 0

    w = p0 + rI * (p1 - p0) - v0
    
    ####
    dir = p1-p0             # // ray direction vector
    intrs = p0+rI*dir#       *I = R.P0 + r * dir;            // intersect point of ray and plane

    ####

    denom = np.inner(u, v) * np.inner(u, v) - \
        np.inner(u, u) * np.inner(v, v)

    si = (np.inner(u, v) * np.inner(w, v) - \
        np.inner(v, v) * np.inner(w, u)) / denom
    
    if (si < 0.0) | (si > 1.0):
        return 0

    ti = (np.inner(u, v) * np.inner(w, u) - \
        np.inner(u, u) * np.inner(w, v)) / denom
    
    if (ti < 0.0) | (si + ti > 1.0):
        return 0

    if (rI == 0.0):
        # point 0 lies ON the triangle. If checking for 
        # point inside polygon, return 2 so that the loop 
        # over triangles can stop, because it is on the 
        # polygon, thus inside.
        return 2
    #axes.scatter(intrs[0],intrs[1], intrs[2],s=100,color='red')

    x=intrs[0]-0.5
    z=intrs[2]-0.5
    
    y=int(intrs[1])
    if(y<0):
        y=y-1
        y_size=(abs(y))*2
        for i in range(y_size):
            position=[(x,y+i,z)]
            size=[(1,1,1)]
            pc=plotCubeAt2(position,size,edgecolor="k")
            axes.add_collection3d(pc)  
    print("Intersection point is x=%f, y=%f, z=%f:" %( intrs[0],intrs[1], intrs[2]))
    return 1


from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Create a new plot
figure = pyplot.figure(figsize=(15,15))
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('sphere.stl')
#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten('F')
axes.auto_scale_xyz(scale, scale, scale)



#these are initial and final points of x and z axis as only y changes
x=np.arange(-40.5,40.5,1)
z=np.arange(-20.5,60.5,1)

#this makes the end points in the graph
xp=np.arange(-40.5,40.5,5)
zp=np.arange(-20.5,60.5,5)
for i in xp:
    for j in zp:
        axes.scatter(i,30,j)


#this is the grid in graph

X=np.arange(-40,45,5) 
Z=np.arange(-20,65,5)
Y=np.array([[-30]])
X,Z=np.meshgrid(X,Z)
axes.plot_wireframe(X,Y,Z,rstride=1, cstride=1)

from itertools import islice
count=0
for i in x:
    for j in z:
        for point in your_mesh.points:
            itr=iter(point)
            t=[list(islice(itr,3)) for i in range(3)]
            t=np.array(t)
            if(ray_intersect_triangle(np.array([i,-25,j]),np.array([i,25,j]),t)==1):
                count+=1
                

#for point in your_mesh.points:
#    itr = iter(point)
#    t=[list(islice(itr,3)) for i in range(3) ]
#    t=np.array(t)
#    for i in x:
#        for j in z:
#            if(ray_intersect_triangle(np.array([i,-20,j]),np.array([i,20,j]),t) == 1):
#                 count+=1 
# Show the plot to the screen

pyplot.show() 

