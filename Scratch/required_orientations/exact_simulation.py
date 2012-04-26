from pylab import *
import rotations
import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure(1)
fig.clear()
ax = fig.add_subplot(111, projection='3d')

class StateError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Circle(object):
    def __init__(self, rot):
        self.rot = array(rot)
        assert abs(norm(self.rot) - 1.) < 1e-5, "Rotation quaternion must have norm 1. Is %g" % norm(self.rot)
        self.intersections = []
        self.intersections_are_sorted = False

    def get_rot(self):
        """Return the rotation as a quaternion"""
        return self.rot

    def __str__(self):
        return "Circle (%g, %g, %g, %g)" % tuple(self.rot)

    def add_intersection(self, intersection, position):
        self.intersections.append((intersection, position))
        self.intersections_are_sorted = False

    def sort_intersections(self):
        self.intersections.sort(key=lambda i: i[1])
        self.intersections_are_sorted = True

    def set_intersection_neighbours(self):
        if not self.intersections_are_sorted:
            raise StateError("Intersections array must be sorted for neighbour search")
        intersections = [i[0] for i in self.intersections]
        array_len = len(intersections)
        for index, intersection in enumerate(intersections):
            neighbour_index_1 = (index+1)%array_len
            neighbour_index_2 = (index-1)%array_len
            # intersection.add_neighbour(intersections[neighbour_index_1])
            # intersection.add_neighbour(intersections[neighbour_index_2])
            intersection.add_neighbours(intersections[neighbour_index_1], intersections[neighbour_index_2])

    def plot(self, color='blue'):
        t = zeros(101)
        t[:100] = arange(0., 2.*pi, 2.*pi/100.)
        t[-1] = 0.
        x = cos(t); y = sin(t); z = zeros(len(x))
        mat = rotations.quaternion_to_matrix(self.rot)
        x_rot = mat[0,0]*x + mat[0,1]*y + mat[0,2]*z
        y_rot = mat[1,0]*x + mat[1,1]*y + mat[1,2]*z
        z_rot = mat[2,0]*x + mat[2,1]*y + mat[2,2]*z
        ax.plot(x_rot, y_rot, z_rot, '-', color=color)

class Intersection(object):
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.neighbours = []
        self.faces = []

    def get_coordinates(self):
        return self.coordinates

    # def add_neighbour(self, neighbour):
    #     self.neighbours.append(neighbour)

    def add_neighbours(self, intersection_1, intersection_2):
        self.neighbours.append([intersection_1, intersection_2])

    def get_neighbours(self, direction):
        return self.neighbours[direction]

    def get_all_neighbours(self):
        return self.neighbours[0] + self.neighbours[1]

    def get_perpendicular_neighbours(self, intersection):
        if intersection in self.neighbours[0]:
            return self.neighbours[1]
        elif intersection in self.neighbours[1]:
            return self.neighbours[0]
        else:
            raise StateError("Intersection is not a neighbour")

    # def get_neighbours(self):
    #     return self.neighbours

    def add_face(self, face):
        self.faces.append(face)

    def number_of_faces(self):
        return len(self.faces)

    def plot(self, color='red'):
        ax.plot([self.coordinates[0]], [self.coordinates[1]], [self.coordinates[2]], 'o', color=color)

def find_intersections(circle_1, circle_2):
    quat_1 = circle_1.get_rot()
    quat_2 = circle_2.get_rot()
    relative_rot_2_to_1 = rotations.quaternion_multiply(quat_1, rotations.quaternion_inverse(quat_2)) #which should be first
    mat = rotations.quaternion_to_matrix(rotations.quaternion_inverse(relative_rot_2_to_1))

    sol = zeros(3)
    sol[1] = mat[2,0]/sqrt(mat[2,0]**2 + mat[2,1]**2)
    sol[0] = -sol[1]*mat[2,1]/mat[2,0]
    intersection_frame_2 = sol
    
    matrix_2 = rotations.quaternion_to_matrix(quat_2) #should probably be 2
    intersection_1 = Intersection(squeeze(array(matrix_2*transpose(matrix(intersection_frame_2)))))
    intersection_2 = Intersection(-intersection_1.get_coordinates())

    mat = rotations.quaternion_to_matrix(rotations.quaternion_inverse(quat_1))

    position_1_1 = math.atan2(*squeeze(array(mat*transpose(matrix(intersection_1.get_coordinates()))))[:2])
    position_1_2 = math.atan2(*squeeze(array(mat*transpose(matrix(intersection_2.get_coordinates()))))[:2])
    
    mat = rotations.quaternion_to_matrix(rotations.quaternion_inverse(quat_2))
    position_2_1 = math.atan2(*squeeze(array(mat*transpose(matrix(intersection_1.get_coordinates()))))[:2])
    position_2_2 = math.atan2(*squeeze(array(mat*transpose(matrix(intersection_2.get_coordinates()))))[:2])

    circle_1.add_intersection(intersection_1, position_1_1)
    circle_1.add_intersection(intersection_2, position_1_2)
    circle_2.add_intersection(intersection_1, position_2_1)
    circle_2.add_intersection(intersection_2, position_2_2)
    
    return intersection_1, intersection_2

def find_all_intersections(circles):
    intersections = []
    for c1, c2 in itertools.combinations(circles, 2):
        this_intersection_1, this_intersection_2 = find_intersections(c1, c2)
        intersections.append(this_intersection_1)
        intersections.append(this_intersection_2)
    return intersections


class Path(object):
    def __init__(self, node_array = None):
        if node_array:
            self.nodes = node_array
        else:
            self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_nodes(self, nodes):
        self.nodes += nodes

    def get_node(self, index):
        try:
            return self.nodes[index]
        except IndexError:
            return None

    def head(self):
        return self.nodes[-1]

    def length(self):
        return len(self.nodes)

    def inverse(self):
        new_path = Path()
        new_path.nodes = self.nodes[::-1]
        return new_path

    def equal(self, path):
        return sorted(path.nodes) == sorted(self.nodes)
    
    def copy(self):
        new_path = Path()
        new_path.nodes = self.nodes[:]
        return new_path

    def __add__(self, path):
        new_path = Path()
        new_path.nodes = self.nodes[:] + path.nodes[:]
        return new_path

    def remove_double(self):
        if self.nodes[0] == self.nodes[-1]:
            self.nodes = self.nodes[:-1]

    def get_center(self):
        pass

    def plot(self, color='black'):
        verts = array([tuple(node.get_coordinates()) for node in self.nodes])
        poly = Poly3DCollection([verts])
        poly.set_color(color)
        poly.set_edgecolor('black')
        ax.add_collection3d(poly)

class RegionFinder(object):
    def __init__(self):
        self.paths = []
        self.regions = []

    def find_region(self, start_intersection):
        #every intersection should be part of four regions
        self.paths = []
        self.regions = []
        
        #setup
        if start_intersection.number_of_faces() >= 4:
            return []
        
        depth_1_intersections = start_intersection.get_neighbours(0) + start_intersection.get_neighbours(1)
        for intersection in depth_1_intersections:
            if intersection.number_of_faces() >= 4:
                depth_1_intersections.remove(intersection)
        depth_1_paths = [Path([start_intersection, intersection]) for intersection in depth_1_intersections]

        #special cases for length 3 and 4 (or only for 3?)
        old_paths = []
        #length 3
        for path_1 in depth_1_paths:
            is_closed = False
            for intersection in path_1.head().get_perpendicular_neighbours(start_intersection):
                if intersection.number_of_faces() >= 4:
                    continue
                if intersection in depth_1_intersections:
                    is_closed = True
                    new_path = path_1.copy()
                    new_path.add_node(intersection)
                    new_is_unique = True
                    for path in self.regions:
                        if path.equal(new_path):
                            new_is_unique = False
                    if new_is_unique:
                        self.regions.append(new_path)
                        for node in self.regions[-1].nodes:
                            node.add_face(self.regions[-1])
                    #break
                else:
                    old_paths.append(path_1.copy())
                    old_paths[-1].add_node(intersection)
            # if not is_closed:
            #     old_paths.append(path_1)
            #     old_paths[-1].add_node(intersection)
        #return self.regions
                    
        # for path_1, path_2 in itertools.combinations(depth_1_paths, 2):            
        #     if path_1.head() in path_2.head().get_perpendicular_neighbours(start_intersection):
        #         self.regions.append(path_1 + path_2.inverse())
        #         self.regions[-1].remove_double()
                
        #         for node in self.regions[-1].nodes:
        #             node.add_face(self.regions[-1])
        
        #look for large loops
        print len(self.regions), " regions found"
        #while len(self.regions) < 4:
        while start_intersection.number_of_faces() < 4:
            print "while restart"
            print [len(path.nodes) for path in old_paths]
            new_paths = []
            #check if endpoints are similar -> new region and paths removed
            #for path_1, path_2 in itertools.combinations(old_paths, 2):
            for path_1 in old_paths:
                is_closed = False
                for path_2 in old_paths:
                    if path_1 != path_2:
                        if path_1.head() == path_2.head():
                            is_closed = True
                            print "close type 1"
                            new_path = path_1.copy()
                            #new_path.nodes += path_2.nodes[1:-1:-1]
                            new_path.nodes += path_2.nodes[-2:0:-1]
                            new_is_unique = True
                            for p in self.regions:
                                if p.equal(new_path):
                                    new_is_unique = False
                            if new_is_unique:
                                print "unique"
                                self.regions.append(new_path)
                                for node in self.regions[-1].nodes:
                                    node.add_face(self.regions[-1])
                if not is_closed:
                    for path_2 in new_paths:
                        #paths will never be identical so no need for testing
                        if path_1.head() == path_2.head():
                            is_closed = True
                            print "close type 2"
                            new_path = path_1.copy()
                            #new_path.nodes += path_2.nodes[1:-1:-1]
                            new_path.nodes += path_2.nodes[-2:0:-1]
                            new_is_unique = True
                            for p in self.regions:
                                if p.equal(new_path):
                                    new_is_unique = False
                            if new_is_unique:
                                print "unique"
                                self.regions.append(new_path)
                                for node in self.regions[-1].nodes:
                                    node.add_face(self.regions[-1])
                            new_paths.remove(path_2)
                if not is_closed:
                    print "don't close:"
                    new_intersections = path_1.head().get_perpendicular_neighbours(path_1.get_node(-2))
                    if (new_intersections[0].number_of_faces() < 4):
                        new_paths.append(path_1.copy())
                        new_paths[-1].add_node(new_intersections[0])
                    if (new_intersections[1].number_of_faces() < 4):
                        new_paths.append(path_1.copy())
                        new_paths[-1].add_node(new_intersections[1])
                    #new_paths += path_1.head().get_perpendicular_neighbours(path_1.get_node(-2))
            old_paths = new_paths

        return self.regions
            
                            
                            

                        
                # if path_1.head() == path_2.head():
                #     new_path = path_1.copy()
                #     new_path.nodes += path_2.nodes[1:-1]
                #     self.regions.append(new_path)
                # else:
                #     new_paths.append(

            #extend paths that are still open and compare the new ones to the created ones on the fly

        #old code
        # depth = 0
        # self.paths.append(Path([start_intersection]))
        # debug_paths = [self.paths]
        # depth += 1
        
        # while len(self.regions) < 4:
        #     old_paths = self.paths
        #     new_paths = []
        #     for path in old_paths:
        #         prev_intersection = path.get_node(-2)
        #         if prev_intersection in path.head().get_neighbours(0):
        #             next_step = path.head().get_neighbours(1)
        #         elif prev_intersection in path.head().get_neighbours(1):
        #             next_step = path.head().get_neighbours(0)
        #         elif prev_intersection == None:
        #             # this should only be true for the first node
        #             print "First node"
        #             next_step = path.head().get_neighbours(0) + path.head().get_neighbours(1)
        #         else:
        #             raise StateError("Previous intersection should be in one of the neighbours")
        #         print "next step = ", next_step
        #         for intersection in next_step:
        #             print "new intersection"
        #             #some if 
        #             new_path = path.copy()
        #             #new_path.add_node(intersection)
        #             old_heads = [p.head() for p in old_paths]
        #             is_closed = False
        #             if intersection in old_heads:
        #                 index = old_heads.index(intersection)
        #                 final_path = new_path + old_paths[index].inverse()
        #                 final_path.remove_double()
        #                 new_is_unique = True
        #                 for path in self.regions:
        #                     if path.equal(final_path):
        #                         new_is_unique = False
        #                 if new_is_unique:
        #                     for node in final_path.nodes:
        #                         node.add_face(final_path)
        #                     self.regions.append(final_path)
        #                 is_closed = True
        #                 #try this
        #                 #old_heads.
        #                 print "is closed"
        #                 #return final_path
        #             new_heads = [p.head() for p in new_paths]
        #             if not is_closed and intersection in new_heads:
        #                 index = new_heads.index(intersection)
        #                 final_path = new_path + new_paths[index].inverse()
        #                 final_path.remove_double()
        #                 new_is_unique = True
        #                 for path in self.regions:
        #                     if path.equal(final_path):
        #                         new_is_unique = False
        #                 if new_is_unique:
        #                     for node in final_path.nodes:
        #                         node.add_face(final_path)
        #                     self.regions.append(final_path)
        #                 is_closed = True
        #                 print "is closed"
        #                 #return final_path
        #             if not is_closed:
        #                 new_path.add_node(intersection)
        #                 new_paths.append(new_path)
        #                 print "add path"


        #     self.paths = new_paths
        #     debug_paths.append(self.paths)
        #     depth += 1
        #     print depth
        # return self.regions, debug_paths
                        
N = 40
circles = []
for i in range(N):
    circles.append(Circle(rotations.random_quaternion()))

intersections = find_all_intersections(circles)

for c in circles:
    c.sort_intersections()

for c in circles:
    c.set_intersection_neighbours()

#check that every intersection has four neighbours
for i in intersections:
    assert len(i.neighbours) == 2, "Intersection has %d directions. Should be 2." % len(i.neighbours)

# plot circles
for c in circles:
    c.plot()

#plot intersections
for i in intersections:
    i.plot()

finder = RegionFinder()
regions = []
for i in range(len(intersections)):
    if intersections[i].number_of_faces() < 4:
        print "** use intersection"
        regions += finder.find_region(intersections[i])
colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'pink', 'gray', 'cyan'] + ['yellow']*100

for region in regions:
    #region.plot(colors[regions.index(region)])
    #region.plot('yellow')
    region.plot(matplotlib.colors.cnames.keys()[random_integers(100)])

ax.set_xlim((-1., 1.))
ax.set_ylim((-1., 1.))
ax.set_zlim((-1., 1.))
draw()
