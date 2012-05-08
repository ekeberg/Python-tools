from pylab import *
import rotations
import itertools
import math
import parallel
import pickle

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
        self.sorted_neighbours = []
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
        """Return the two neighbours that are perpendicular to the input"""
        if intersection in self.neighbours[0]:
            return self.neighbours[1]
        elif intersection in self.neighbours[1]:
            return self.neighbours[0]
        else:
            raise StateError("Intersection is not a neighbour")

    def sort_neighbours(self):
        """Creates sorted_neighbours that orders the neighbours in a geometrical order"""
        self.sorted_neighbours.append(self.neighbours[0][0])
        z = self.coordinates
        # project n1 on the tangent plane of the sphere (coordinate system is recentered on self.coordinates)
        n1_proj = (self.sorted_neighbours[0].coordinates -
                   self.coordinates*dot(self.coordinates, self.sorted_neighbours[0].coordinates) /
                   norm(self.coordinates)**2)
        # calculate basis vectors 
        x = n1_proj / norm(n1_proj)
        y = cross(z,x)
        n = self.neighbours[1][0].coordinates
        nx = dot((n - self.coordinates), x)
        ny = dot((n - self.coordinates), y)
        # alpha is the angle in the tangent plane relative to neighbour[0][0]
        alpha = math.atan2(ny, nx)
        if alpha > 0.:
            self.sorted_neighbours.append(self.neighbours[1][0])
            self.sorted_neighbours.append(self.neighbours[0][1])
            self.sorted_neighbours.append(self.neighbours[1][1])
        else:
            self.sorted_neighbours.append(self.neighbours[1][1])
            self.sorted_neighbours.append(self.neighbours[0][1])
            self.sorted_neighbours.append(self.neighbours[1][0])

    def get_neighbour_right(self, intersection):
        if not self.sorted_neighbours:
            raise StateError("Function get_neighbour_right called before sorted_neighbours were initialized")
        index = self.sorted_neighbours.index(intersection)
        new_index = (index-1)%4
        return self.sorted_neighbours[new_index]
    
    def get_neighbour_left(self, intersection):
        if not self.sorted_neighbours:
            raise StateError("Function get_neighbour_right called before sorted_neighbours were initialized")
        index = self.sorted_neighbours.index(intersection)
        new_index = (index+1)%4
        return self.sorted_neighbours[new_index]

    def get_sorted_neighbours(self):
        return self.sorted_neighbours

    # def get_neighbours(self):
    #     return self.neighbours

    def add_face(self, face):
        self.faces.append(face)

    def number_of_faces(self):
        return len(self.faces)

    def get_faces(self):
        return self.faces

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

    def get_nodes(self):
        return self.nodes

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

    def order_independent_hash(self):
        """Hash calculated from the nodes contained"""
        sorted_nodes = sorted(self.nodes)
        return hash(str(sorted_nodes))

    def get_gap_size(self):
        # project corners on most even plane (pca)
        coordinates = array([n.coordinates for n in self.nodes])
        coordinates_avg = average(coordinates, axis=0)
        coordinates -= coordinates_avg
        # do a svd on the coordinates, transpose to get coordinates in columns.
        U, S, Vh = svd(coordinates.T)
        # project coordinates on principal coordinates
        Y = dot(U.T, coordinates.T)
        projected = Y[:2]
        # run algorithm

        def get_center_from_3_lines(lines):
            m = matrix([[lines[0,1,1] - lines[0,0,1], lines[0,0,0] - lines[0,1,0],
                         -sqrt((lines[0,1,0] - lines[0,0,0])**2 + (lines[0,1,1] - lines[0,0,1])**2)],
                        [lines[1,1,1] - lines[1,0,1], lines[1,0,0] - lines[1,1,0],
                         -sqrt((lines[1,1,0] - lines[1,0,0])**2 + (lines[1,1,1] - lines[1,0,1])**2)],
                        [lines[2,1,1] - lines[2,0,1], lines[2,0,0] - lines[2,1,0],
                         -sqrt((lines[2,1,0] - lines[2,0,0])**2 + (lines[2,1,1] - lines[2,0,1])**2)]])
            d = matrix([[lines[0,0,0] * (lines[0,1,1] - lines[0,0,1]) - lines[0,0,1] * (lines[0,1,0] - lines[0,0,0])],
                        [lines[1,0,0] * (lines[1,1,1] - lines[1,0,1]) - lines[1,0,1] * (lines[1,1,0] - lines[1,0,0])],
                        [lines[2,0,0] * (lines[2,1,1] - lines[2,0,1]) - lines[2,0,1] * (lines[2,1,0] - lines[2,0,0])]])
            try:
                solution = solve(m,d)
            except LinAlgError:
                # seems like the paths are sometimes traced twice, or wrong at least
                print "coordinates = ", coordinates
                print "projected = ", projected
                print "projected_permuted = ", projected_permuted
                print "lines = ", lines
                print "all_lines = ", all_lines
                print "m = ", m
                print "d = ", d
                raise LinAlgError("Singular matrix")

            solution = squeeze(array(solution))
            c = solution[:2]
            d = solution[2]

            # project center on the lines

            projected_point_1 = (lines[0,0] + dot((c - lines[0,0]), (lines[0,1] - lines[0,0])) /
                                 norm(lines[0,1] - lines[0,0])**2 * (lines[0,1] - lines[0,0]))
            projected_point_2 = (lines[1,0] + dot((c - lines[1,0]), (lines[1,1] - lines[1,0])) /
                                 norm(lines[1,1] - lines[1,0])**2 * (lines[1,1] - lines[1,0]))
            projected_point_3 = (lines[2,0] + dot((c - lines[2,0]), (lines[2,1] - lines[2,0])) /
                                 norm(lines[2,1] - lines[2,0])**2 * (lines[2,1] - lines[2,0]))

            angle_1 = math.atan2(*(projected_point_1-c)[::-1])
            angle_2 = math.atan2(*(projected_point_2-c)[::-1])
            angle_3 = math.atan2(*(projected_point_3-c)[::-1])

            def mod_dist(a, b):
                return min(abs(a-b), abs(a-b+2.*pi), abs(a-b-2.*pi))

            # print "angles = (%g, %g, %g)" % (angle_1, angle_2, angle_3)
            # print "1->2: %g, 2->3: %g, 3->1: %g" % (mod_dist(angle_1, angle_2),
            #                                         mod_dist(angle_2, angle_3),
            #                                         mod_dist(angle_3, angle_1))
            if mod_dist(angle_1, angle_2) + mod_dist(angle_2, angle_3) + mod_dist(angle_3, angle_1) > (2.*pi - 0.001):
                is_closed = True
            else:
                is_closed = False

            #Get angle of vector from center to projected center (using atan2)

            #if there is a gap of pi it is open
            #test this by summing the distances between the angles. == 2 pi -> closed < 2 pi -> open

            return c, d, is_closed

        projected_permuted = projected.copy()
        projected_permuted[:,:-1] = projected[:,1:]
        projected_permuted[:,-1] = projected[:,0]
        all_lines = array(zip(list(projected.T), list(projected_permuted.T)))
        #lines = array([[list(point0) for point0 in line0]  for line0 in lines])
        all_lines = [l for l in all_lines]

        centers_distances_closed = []
        for line_1, line_2, line_3 in itertools.combinations(all_lines, 3):
            c, d, closed = get_center_from_3_lines(array([line_1, line_2, line_3]))
            centers_distances_closed.append((c, d, closed, (line_1, line_2, line_3)))

        centers_distances = [i for i in centers_distances_closed if i[2]]
        best_c_and_d = min(centers_distances, key=lambda x: abs(x[1]))

        # transform center back to the original coordinate system
        center = dot(U, array(list(best_c_and_d[0])+[0.])) + coordinates_avg
        
        #ax.plot(best_c_and_d[0][0], best_c_and_d[0][1], 'o', color='red')
        #limiting_lines = array(best_c_and_d[3])

        return best_c_and_d[1], center

    def plot(self, color='black'):
        verts = array([tuple(node.get_coordinates()) for node in self.nodes])
        poly = Poly3DCollection([verts])
        poly.set_color(color)
        poly.set_edgecolor('black')
        ax.add_collection3d(poly)

def path_sanity_check(path):
    """check that neighbours in path are true neighbours"""
    for i in range(path.length()):
        if not (path.nodes[i] in path.nodes[(i+1)%path.length()].get_all_neighbours()):
            raise AssertionError("path has no-neighbour connection")
        if not (path.nodes[i] in path.nodes[(i+path.length()-1)%path.length()].get_all_neighbours()):
            raise AssertionError("path has no-neighbour connection")

class RegionFinder(object):
    def __init__(self):
        self.paths = []
        self.all_regions = []

    def find_region_with_sorted_paths(self, start_intersection):
        regions = []
        for second_intersection in start_intersection.get_sorted_neighbours():
            new_path = Path([start_intersection, second_intersection])
            intersection = second_intersection.get_neighbour_right(start_intersection)
            while intersection != start_intersection:
                new_path.add_node(intersection)
                intersection = new_path.head().get_neighbour_right(new_path.get_node(-2))
            #if not new_path.nodes in [face.nodes for face in start_intersection.get_faces()]:
            if not new_path.order_independent_hash() in [face.order_independent_hash() for face in start_intersection.get_faces()]:
                regions.append(new_path)
                for node in new_path.get_nodes():
                    node.add_face(new_path)
        return regions

    def find_region(self, start_intersection):
        return self.find_region_with_sorted_paths(start_intersection)

    def find_region_old(self, start_intersection):
        print " "
        print "Start function"
        print "faces around = %d" % start_intersection.number_of_faces()
        print "neighbouring faces around = ", [i.number_of_faces() for i in start_intersection.get_all_neighbours()]
        #every intersection should be part of four regions
        self.paths = []
        self.regions = []
        # doesn't work yet!
        # idea is to keep track of nodes that have been passed so that we don't trace a path through the central
        # part. This doesn't completely solve the problem of tracing non-empty paths however.
        seen_nodes = [start_intersection]
        
        #setup
        # if this intersection is already included in four faces there is nothing for us to do.
        if start_intersection.number_of_faces() >= 4:
            return []

        # the neighbouring intersections are listed and the ones already present
        # in four faces are removed since they are not interesting.
        depth_1_intersections = start_intersection.get_neighbours(0) + start_intersection.get_neighbours(1)
        for intersection in depth_1_intersections:
            if intersection.number_of_faces() >= 4:
                depth_1_intersections.remove(intersection)
        depth_1_paths = [Path([start_intersection, intersection]) for intersection in depth_1_intersections]
        seen_nodes += depth_1_intersections

        #special cases for length 3 and 4 (or only for 3?)
        old_paths = []
        #length 3
        for path_1 in depth_1_paths:
            is_closed = False
            for intersection in path_1.head().get_perpendicular_neighbours(start_intersection):
                if intersection.number_of_faces() >= 4:
                    print "continue - %d" % intersection.number_of_faces()
                    continue
                if intersection in depth_1_intersections:
                    print "closed"
                    is_closed = True
                    new_path = path_1.copy()
                    new_path.add_node(intersection)
                    path_sanity_check(new_path)
                    new_is_unique = True
                    for path in self.regions:
                        if path.equal(new_path):
                            new_is_unique = False
                    if new_is_unique:
                        self.regions.append(new_path)
                        for node in self.regions[-1].nodes:
                            if node.number_of_faces() >= 4:
                                print "first"
                                print node.faces
                                raise AssertionError("Intersection is part of more than four faces")
                            node.add_face(self.regions[-1])
                    #break
                else:
                    print "new"
                    old_paths.append(path_1.copy())
                    old_paths[-1].add_node(intersection)
                seen_nodes.append(intersection)

        print "old_paths length = %d" % len(old_paths)
        # at this point we sould have length 3 paths in old_paths. They might
        # have common heads (two paths together might form a closed loop) but
        # except for this no parts will overlap
        
        #look for large loops
        #print len(self.regions), " regions found"
        #while len(self.regions) < 4:
        print "start while"
        while start_intersection.number_of_faces() < 4:
            print len(old_paths), ", ", start_intersection.number_of_faces()
            assert len(old_paths) > 0, "No paths in old paths to analyze even though all paths are not found"
            new_paths = []
            #check if endpoints are similar -> new region and paths removed
            #for path_1, path_2 in itertools.combinations(old_paths, 2):
            for path_1 in old_paths:
                is_closed = False
                for path_2 in old_paths:
                    if path_1 != path_2:
                        if path_1.head() == path_2.head():
                            is_closed = True
                            #print "close type 1"
                            new_path = path_1.copy()
                            #new_path.nodes += path_2.nodes[1:-1:-1]
                            new_path.nodes += path_2.nodes[-2:0:-1]
                            path_sanity_check(new_path)
                            new_is_unique = True
                            for p in self.regions:
                                if p.equal(new_path):
                                    new_is_unique = False
                            if new_is_unique:
                                #print "unique"
                                self.regions.append(new_path)
                                for node in self.regions[-1].nodes:
                                    if node.number_of_faces() >= 4:
                                        print "second"
                                        print node.faces
                                        raise AssertionError("Intersection is part of more than four faces")
                                    node.add_face(self.regions[-1])
                if not is_closed:
                    for path_2 in new_paths:
                        #paths will never be identical so no need for testing
                        if path_1.head() == path_2.head():
                            is_closed = True
                            #print "close type 2"
                            new_path = path_1.copy()
                            #new_path.nodes += path_2.nodes[1:-1:-1]
                            new_path.nodes += path_2.nodes[-2:0:-1]
                            path_sanity_check(new_path)
                            new_is_unique = True
                            for p in self.regions:
                                if p.equal(new_path):
                                    new_is_unique = False
                            if new_is_unique:
                                #print "unique"
                                self.regions.append(new_path)
                                for node in self.regions[-1].nodes:
                                    if node.number_of_faces() >= 4:
                                        print "third"
                                        print node.faces
                                        raise AssertionError("Intersection is part of more than four faces")
                                    node.add_face(self.regions[-1])
                            new_paths.remove(path_2)
                if not is_closed:
                    #print "don't close:"
                    new_intersections = path_1.head().get_perpendicular_neighbours(path_1.get_node(-2))
                    if not (new_intersections[0] in seen_nodes):
                        if (new_intersections[0].number_of_faces() < 4):
                            new_paths.append(path_1.copy())
                            new_paths[-1].add_node(new_intersections[0])
                            seen_nodes.append(new_intersections[0])
                    if not (new_intersections[1] in seen_nodes):
                        if (new_intersections[1].number_of_faces() < 4):
                            new_paths.append(path_1.copy())
                            new_paths[-1].add_node(new_intersections[1])
                            seen_nodes.append(new_intersections[1])

                    #new_paths += path_1.head().get_perpendicular_neighbours(path_1.get_node(-2))
            old_paths = new_paths

        #assert len(old_paths) == 0, ("Ending path searching even though old paths is not empty." + str(old_paths))
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


#N = 5
def gap_size_from_N(N):
    seed()
    circles = []
    for i in range(N):
        circles.append(Circle(rotations.random_quaternion()))

    intersections = find_all_intersections(circles)

    for c in circles:
        c.sort_intersections()

    for c in circles:
        c.set_intersection_neighbours()

    for i in intersections:
        i.sort_neighbours()

    for i in intersections:
        assert len(i.neighbours) == 2, "intersection has %d directions. Should be 2" % len(i.neighbours)

    finder = RegionFinder()
    regions = []
    for i in range(len(intersections)):
        if intersections[i].number_of_faces() < 4:
            regions += finder.find_region(intersections[i])

    for i in intersections:
        assert i.number_of_faces() == 4, "Intersection %d is part of %d faces" % (intersections.index(i),
                                                                                  i.number_of_faces())


    largest_region = max(regions, key=lambda x: x.get_gap_size()[0])
    return largest_region.get_gap_size()[0]

# for c in circles:
#     c.plot()
# for r in regions:
#     r.plot(matplotlib.colors.cnames.keys()[random_integers(100)])
    
# ax.set_xlim((-1, 1))
# ax.set_ylim((-1, 1))
# ax.set_zlim((-1, 1))
# draw()

class Gaps(object):
    def __init__(self, gaps, number_of_images, doc=None):
        self._gaps = gaps
        if doc: self._doc = doc
        self._number_of_images = number_of_images

    def gaps(self):
        return self._gaps

    def doc(self):
        return self._doc

    def number_of_images(self):
        return self._number_of_images

N_list = range(10, 601, 10)
number_of_repetitions = 100
def calculate_gaps(N_list, number_of_repetitions, filename):
    """Pickle the result to file"""
    gap_average = []
    gap_std = []
    gap_median = []
    gaps_all = []
    for N in N_list:
        #gaps = []
        jobs = ((N,),)*number_of_repetitions
        gaps = parallel.run_parallel(jobs, gap_size_from_N, quiet=True)
        # for i in range(number_of_repetiotions):
        #     gaps.append(gap_gap_from_N(N))
        #     print "gap_gap = %g" % gaps[-1]
        gap_average.append(average(gaps))
        gap_std.append(std(gaps))
        gap_median.append(median(gaps))
        gaps_all.append(gaps)
        print "(N = %d) average = %g, std = %g" % (N, gap_average[-1], gap_std[-1])
        
        gap_out = Gaps(array(gaps_all), N_list)
        file_handle = open(filename, 'wb')
        pickle.dump(gap_out, file_handle)
        file_handle.close()

    #return gap_average, gap_std, gap_median, array(gaps_all)

#gap_average, gap_std, gap_median, gaps_all = calculate_gaps(N_list, number_of_repetitions)

# translate gaps to nyquist pixels

resolution_list = [10., 15., 20., 25., 30.] #This is the number of resolution elments along the object
def plot_relolutions(resolution_list, gaps_all, N_list):
    full_coverage_plot = []
    for resolution in resolution_list:
        conversion_factor = resolution/2. # nyquist = coordinate * conversion_factor
        gaps_nyquist_all = gaps_all * conversion_factor
        full_coverage_all = gaps_nyquist_all <= 1.
        full_coverage_probability = average(full_coverage_all, axis=1)
        full_coverage_plot.append(full_coverage_probability)

    clf()
    for i, p in enumerate(full_coverage_plot):
        plot(N_list, p, label=str(resolution_list[i]))
    legend()
    draw()
# calculate for which gap sizes full coverage is reached

# N = 30
# circles = []
# for i in range(N):
#     circles.append(Circle(rotations.random_quaternion()))

# intersections = find_all_intersections(circles)

# for c in circles:
#     c.sort_intersections()

# for c in circles:
#     c.set_intersection_neighbours()

# #check that every intersection has four neighbours
# for i in intersections:
#     assert len(i.neighbours) == 2, "Intersection has %d directions. Should be 2." % len(i.neighbours)

# # plot circles
# for c in circles:
#     c.plot()

# #plot intersections
# for i in intersections:
#     i.plot()

# finder = RegionFinder()
# regions = []
# for i in range(len(intersections)):
#     if intersections[i].number_of_faces() < 4:
#         print "** use intersection"
#         regions += finder.find_region(intersections[i])
# colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'pink', 'gray', 'cyan'] + ['yellow']*100

# for region in regions:
#     #region.plot(colors[regions.index(region)])
#     #region.plot('yellow')
#     region.plot(matplotlib.colors.cnames.keys()[random_integers(100)])
    

# for region in regions:
#     d, c = region.get_gap_size()
#     ax.plot3D([c[0]], [c[1]], [c[2]], 'o', color='black', zorder=1000)
# region_sizes = [region.get_gap_size() for region in regions]
# largest_region = max(regions, key=lambda x: x.get_gap_size()[0])
# print "largest gap = %g" % largest_region.get_gap_size()[0]

# ax.set_xlim((-1., 1.))
# ax.set_ylim((-1., 1.))
# ax.set_zlim((-1., 1.))
# draw()


