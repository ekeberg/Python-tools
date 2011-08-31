import pylab
import h5py
from MarchingCubesResources import *

def read_map(side):
    filename = '/Users/ekeberg/Work/scratch/mimi3d_image1.h5'
    #filename = '/Users/ekeberg/Work/scratch/mimi3d-avg_image.h5'
    f = h5py.File(filename)
    i = f.keys().index('real')
    image = abs(f.values()[i].value)

    image = pylab.fftshift(image)

    image[image < 0.] = 0.0

    small_image_side = 12*2

    center = [62,62,62]
    #center = [78,82,56]

    image_center = image[center[0]-12:center[0]+12,center[1]-12:center[1]+12,center[2]-12:center[2]+12]

    image_ft = pylab.fftn(image_center,[24,24,24])
    image_big = abs(pylab.ifftn(pylab.fftshift(image_ft),[side,side,side]))

    image_big /= max(image_big.flatten())

    return image_big

side = 256
density_map = read_map(side)

#threshold = pylab.array([0.25,0.4,0.55,0.7,0.75])*max(density_map.flatten())
threshold = pylab.array([0.25,0.4,0.55,0.7,0.75])[3]*max(density_map.flatten())

corners = density_map > threshold


cubes = []
for i in range(pylab.shape(density_map)[0]-1):
    for j in range(pylab.shape(density_map)[1]-1):
        for k in range(pylab.shape(density_map)[2]-1):
            cubes.append((corners[i:i+2,j:j+2,k:k+2],(i,j,k),density_map[i:i+2,j:j+2,k:k+2]))

vertices = []
triangle_indices = []
triangles = []
cube_edges = -1*pylab.ones(pylab.shape(density_map)+(3,),dtype='int32')
print "%d cubes" % len(cubes)
for c,c_i in zip(cubes,range(len(cubes))):
    if c_i % int(len(cubes)*0.1) == 0:
        print "%d cubes processed" % c_i
    lookup = 0
    				# // 7 -- x + y*this->size_y + z * this->size_y * this->size_z
				# if (this->vertices[idx].inside) lookup |= 128;
				# // 6 -- (x + 1) + y*this->size_y + z * this->size_y * this->size_z
				# if (this->vertices[idx+1].inside) lookup |= 64;
				# // 2 -- (x + 1) + (y + 1)*this->size_y + z * this->size_y * this->size_z
				# if (this->vertices[idx+1+this->size_y].inside) lookup |= 4;
				# // 3 -- x + (y + 1)*this->size_y + z * this->size_y * this->size_z
				# if (this->vertices[idx + this->size_y].inside) lookup |= 8;
				# // 4 -- x + y*this->size_y + (z + 1) * this->size_y * this->size_z
				# if (this->vertices[idx + (this->size_y * this->size_z)].inside) lookup |= 16;
				# // 5 -- (x + 1) + y*this->size_y + (z + 1) * this->size_y * this->size_z
				# if (this->vertices[idx + 1 + (this->size_y * this->size_z)].inside) lookup |= 32;
				# // 1 -- (x + 1) + (y + 1)*this->size_y + (z + 1) * this->size_y * this->size_z
				# if (this->vertices[idx + 1 + this->size_y + (this->size_y * this->size_z)].inside) lookup |= 2;
				# // 0 -- x + (y + 1)*this->size_y + (z + 1) * this->size_y * this->size_z
				# if (this->vertices[idx + this->size_y + (this->size_y * this->size_z)].inside) lookup |= 1;

    # if c[0][0,0,0]: lookup |= 128
    # if c[0][0,1,0]: lookup |= 64
    # if c[0][0,1,1]: lookup |= 32
    # if c[0][0,0,1]: lookup |= 16
    # if c[0][1,0,0]: lookup |= 8
    # if c[0][1,1,0]: lookup |= 4
    # if c[0][1,1,1]: lookup |= 2
    # if c[0][1,0,1]: lookup |= 1
    if c[0][0,0,0]: lookup |= 128
    if c[0][1,0,0]: lookup |= 64
    if c[0][1,0,1]: lookup |= 32
    if c[0][0,0,1]: lookup |= 16
    if c[0][0,1,0]: lookup |= 8
    if c[0][1,1,0]: lookup |= 4
    if c[0][1,1,1]: lookup |= 2
    if c[0][0,1,1]: lookup |= 1
    if lookup != 0 or lookup != 255:
        verts = [0,0,0,0,0,0,0,0,0,0,0,0]
        # if edgeTable[lookup] & 1: verts[0] = ((c[1][0]+0.5,c[1][1]+1.0,c[1][2]+1.0))
        # if edgeTable[lookup] & 2: verts[1] = ((c[1][0]+1.0,c[1][1]+1.0,c[1][2]+0.5))
        # if edgeTable[lookup] & 4: verts[2] = ((c[1][0]+0.5,c[1][1]+1.0,c[1][2]+0.0))
        # if edgeTable[lookup] & 8: verts[3] = ((c[1][0]+0.0,c[1][1]+1.0,c[1][2]+0.5))
        # if edgeTable[lookup] & 16: verts[4] = ((c[1][0]+0.5,c[1][1]+0.0,c[1][2]+1.0))
        # if edgeTable[lookup] & 32: verts[5] = ((c[1][0]+1.0,c[1][1]+0.0,c[1][2]+0.5))
        # if edgeTable[lookup] & 64: verts[6] = ((c[1][0]+0.5,c[1][1]+0.0,c[1][2]+0.0))
        # if edgeTable[lookup] & 128: verts[7] = ((c[1][0]+0.0,c[1][1]+0.0,c[1][2]+0.5))
        # if edgeTable[lookup] & 256: verts[8] = ((c[1][0]+0.0,c[1][1]+0.5,c[1][2]+1.0))
        # if edgeTable[lookup] & 512: verts[9] = ((c[1][0]+1.0,c[1][1]+0.5,c[1][2]+1.0))
        # if edgeTable[lookup] & 1024: verts[10] = ((c[1][0]+1.0,c[1][1]+0.5,c[1][2]+0.0))
        # if edgeTable[lookup] & 2048: verts[11] = ((c[1][0]+0.0,c[1][1]+0.5,c[1][2]+0.0))
        if edgeTable[lookup] & 1: verts[0] = ((c[1][0]+(threshold-c[2][0,1,1])/(c[2][1,1,1]-c[2][0,1,1]),c[1][1]+1.0,c[1][2]+1.0))
        if edgeTable[lookup] & 2: verts[1] = ((c[1][0]+1.0,c[1][1]+1.0,c[1][2]+(threshold-c[2][1,1,0])/(c[2][1,1,1]-c[2][1,1,0])))
        if edgeTable[lookup] & 4: verts[2] = ((c[1][0]+(threshold-c[2][0,1,0])/(c[2][1,1,0]-c[2][0,1,0]),c[1][1]+1.0,c[1][2]+0.0))
        if edgeTable[lookup] & 8: verts[3] = ((c[1][0]+0.0,c[1][1]+1.0,c[1][2]+(threshold-c[2][0,1,0])/(c[2][0,1,1]-c[2][0,1,0])))
        if edgeTable[lookup] & 16: verts[4] = ((c[1][0]+(threshold-c[2][0,0,1])/(c[2][1,0,1]-c[2][0,0,1]),c[1][1]+0.0,c[1][2]+1.0))
        if edgeTable[lookup] & 32: verts[5] = ((c[1][0]+1.0,c[1][1]+0.0,c[1][2]+(threshold-c[2][1,0,0])/(c[2][1,0,1]-c[2][1,0,0])))
        if edgeTable[lookup] & 64: verts[6] = ((c[1][0]+(threshold-c[2][0,0,0])/(c[2][1,0,0]-c[2][0,0,0]),c[1][1]+0.0,c[1][2]+0.0))
        if edgeTable[lookup] & 128: verts[7] = ((c[1][0]+0.0,c[1][1]+0.0,c[1][2]+(threshold-c[2][0,0,0])/(c[2][0,0,1]-c[2][0,0,0])))
        if edgeTable[lookup] & 256: verts[8] = ((c[1][0]+0.0,c[1][1]+(threshold-c[2][0,0,1])/(c[2][0,1,1]-c[2][0,0,1]),c[1][2]+1.0))
        if edgeTable[lookup] & 512: verts[9] = ((c[1][0]+1.0,c[1][1]+(threshold-c[2][1,0,1])/(c[2][1,1,1]-c[2][1,0,1]),c[1][2]+1.0))
        if edgeTable[lookup] & 1024: verts[10] = ((c[1][0]+1.0,c[1][1]+(threshold-c[2][1,0,0])/(c[2][1,1,0]-c[2][1,0,0]),c[1][2]+0.0))
        if edgeTable[lookup] & 2048: verts[11] = ((c[1][0]+0.0,c[1][1]+(threshold-c[2][0,0,0])/(c[2][0,1,0]-c[2][0,0,0]),c[1][2]+0.0))

        tri = triTable[lookup]
        tri_stripped = [t for t in tri if t != -1]
        # for t in range(len(tri_stripped)/3):
        #     triangles.append((verts[tri_stripped[3*t]],verts[tri_stripped[3*t+1]],verts[tri_stripped[3*t+2]]))

        for t in range(len(tri_stripped)/3):
            this_triangle = []
            for s in tri_stripped[3*t:3*t+3]:
                if s == 0: coord = (0,1,1,0)
                if s == 1: coord = (1,1,0,2)
                if s == 2: coord = (0,1,0,0)
                if s == 3: coord = (0,1,0,2)
                if s == 4: coord = (0,0,1,0)
                if s == 5: coord = (1,0,0,2)
                if s == 6: coord = (0,0,0,0)
                if s == 7: coord = (0,0,0,2)
                if s == 8: coord = (0,0,1,1)
                if s == 9: coord = (1,0,1,1)
                if s == 10: coord = (1,0,0,1)
                if s == 11: coord = (0,0,0,1)
                coord = tuple(pylab.array(coord) + pylab.array(c[1]+(0,)))

                if cube_edges[coord] == -1:
                    vertices.append(verts[s])
                    this_triangle.append(len(vertices)-1)
                    cube_edges[coord] = len(vertices)-1
                else:
                    this_triangle.append(cube_edges[coord])

            triangle_indices.append((this_triangle[0],this_triangle[1],this_triangle[2]))
                    

print "start output"
f = open('isosurface.data','wp')
f.write("%d %d\n" % (len(vertices), len(triangle_indices)))
for i in range(len(vertices)):
    for j in range(3):
        f.write(str(vertices[i][j])+' ')
    f.write('\n')
for i in range(len(triangle_indices)):
    for j in range(3):
        f.write(str(triangle_indices[i][j])+' ')
    f.write('\n')
f.close()
print "done"


doPlot = False
if doPlot:
    print "start plotting"
    from mpl_toolkits.mplot3d import axes3d
    fig = pylab.figure(1)
    fig.clear()
    ax = axes3d.Axes3D(fig)
    t = pylab.array(vertices)

    #ax.plot3D(t_center[:,0],t_center[:,1],t_center[:,2],'o')
    # t_wrap = pylab.zeros((pylab.shape(t)[0],4,3))
    # t_wrap[:,:3,:] = t[:,:,:]
    # t_wrap[:,3,:] = t[:,0,:]
    for i in range(len(triangle_indices)):
        this_triangle = [triangle_indices[i][0],triangle_indices[i][1],triangle_indices[i][2],triangle_indices[i][0]]
        ax.plot3D(t[this_triangle,0],t[this_triangle,1],t[this_triangle,2])
        #ax.plot3D(t_wrap[i,:,0],t_wrap[i,:,1],t_wrap[i,:,2])
        #ax.plot3D(t_wrap[0,:,0],t_wrap[0,:,1],t_wrap[0,:,2],lw=4)
    pylab.show()
    print "done"
else:
    print "start plotting"
    from mpl_toolkits.mplot3d import axes3d
    fig = pylab.figure(1)
    fig.clear()
    ax = axes3d.Axes3D(fig)
    t = pylab.array(vertices)

    #for i in range(pylab.shape(vertices)[0]):
    ax.plot3D(t[:,0],t[:,1],t[:,2],'o')
    pylab.show()
    print "done"
    

