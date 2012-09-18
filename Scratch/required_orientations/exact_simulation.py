"""This module is used to simulate the size of the gaps in fourier space expected after
incorporating a certain number of images at random orientation"""

#from pylab import *
import pylab
from pylab import pi
import rotations
import itertools
import math
import parallel
import pickle
import time_tools


class StateError(Exception):
    "Error used to describe that something is executed out of order."
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Circle(object):
    """Randomly oriented geodesic circle"""
    def __init__(self, rot):
        self._rot = pylab.array(rot)
        assert abs(pylab.norm(self._rot) - 1.) < 1e-5, \
               "Rotation quaternion must have norm 1. Is %g" % pylab.norm(self._rot)
        self._intersections = []
        self._intersections_are_sorted = False

    def get_rot(self):
        """Return the rotation as a quaternion"""
        return self._rot

    def __str__(self):
        return "Circle (%g, %g, %g, %g)" % tuple(self._rot)

    def add_intersection(self, intersection, position):
        """Add an intersection at a certain angle (position) of the circle"""
        self._intersections.append((intersection, position))
        self._intersections_are_sorted = False

    def sort_intersections(self):
        """Sort intersections so that neighborus are true neighbours"""
        self._intersections.sort(key=lambda i: i[1])
        self._intersections_are_sorted = True

    def set_intersection_neighbours(self):
        """For all intersections along the circle, make them know about their neighbours along this circle"""
        if not self._intersections_are_sorted:
            raise StateError("Intersections array must be sorted for neighbour search")
        intersections = [i[0] for i in self._intersections]
        array_len = len(intersections)
        for index, intersection in enumerate(intersections):
            neighbour_index_1 = (index+1)%array_len
            neighbour_index_2 = (index-1)%array_len
            # intersection.add_neighbour(intersections[neighbour_index_1])
            # intersection.add_neighbour(intersections[neighbour_index_2])
            intersection.add_neighbours(intersections[neighbour_index_1], intersections[neighbour_index_2])

    def plot(self, axis, color='blue'):
        """Plot the circle"""
        circle_parametrization = pylab.zeros(101)
        circle_parametrization[:100] = pylab.arange(0., 2.*pi, 2.*pi/100.)
        circle_parametrization[-1] = 0.
        segment_x = pylab.cos(circle_parametrization)
        segment_y = pylab.sin(circle_parametrization)
        segment_z = pylab.zeros(len(segment_x))
        mat = rotations.quaternion_to_matrix(self._rot)
        rotated_x = mat[0, 0]*segment_x + mat[0, 1]*segment_y + mat[0, 2]*segment_z
        rotated_y = mat[1, 0]*segment_x + mat[1, 1]*segment_y + mat[1, 2]*segment_z
        rotated_z = mat[2, 0]*segment_x + mat[2, 1]*segment_y + mat[2, 2]*segment_z
        axis.plot(rotated_x, rotated_y, rotated_z, '-', color=color)

class Intersection(object):
    """An intersection between two Circle objects"""
    def __init__(self, coordinates):
        self._coordinates = coordinates
        self._neighbours = []
        self._sorted_neighbours = []
        self._faces = []

    def get_coordinates(self):
        """The 3D coordinates of the intersection"""
        return self._coordinates

    # def add_neighbour(self, neighbour):
    #     self.neighbours.append(neighbour)

    def add_neighbours(self, intersection_1, intersection_2):
        """Add a pair of neighbours. The two neighbours are assumed to
        lie on the sime line, and therefore be opposite"""
        self._neighbours.append([intersection_1, intersection_2])

    def get_neighbours(self, direction=None):
        """Return a neighbour pair. Valid input is 0 and 1. Or if no input is given
        return both the neighbour pairs"""
        if direction:
            return self._neighbours[direction]
        else:
            return self._neighbours

    def get_all_neighbours(self):
        """return all (four) neighbours as one array"""
        return self._neighbours[0] + self._neighbours[1]

    def get_perpendicular_neighbours(self, intersection):
        """Return the two neighbours that are perpendicular to the input"""
        if intersection in self._neighbours[0]:
            return self._neighbours[1]
        elif intersection in self._neighbours[1]:
            return self._neighbours[0]
        else:
            raise StateError("Intersection is not a neighbour")

    def sort_neighbours(self):
        """Creates sorted_neighbours that orders the neighbours in a geometrical order"""
        self._sorted_neighbours.append(self._neighbours[0][0])
        new_basis_z = self._coordinates
        # project n1 on the tangent plane of the sphere (coordinate system is recentered on self.coordinates)
        n1_proj = (self._sorted_neighbours[0].get_coordinates() -
                   self._coordinates*pylab.dot(self._coordinates,
                                               self._sorted_neighbours[0].get_coordinates()) /
                   pylab.norm(self.get_coordinates())**2)
        # calculate basis vectors 
        new_basis_x = n1_proj / pylab.norm(n1_proj)
        new_basis_y = pylab.cross(new_basis_z, new_basis_x)
        neighbour_1 = self._neighbours[1][0].get_coordinates()
        neighbour_1_new_x = pylab.dot((neighbour_1 - self._coordinates), new_basis_x)
        neighbour_1_new_y = pylab.dot((neighbour_1 - self._coordinates), new_basis_y) 
        # alpha is the angle in the tangent plane relative to neighbour[0][0]
        alpha = math.atan2(neighbour_1_new_y, neighbour_1_new_x)
        if alpha > 0.:
            self._sorted_neighbours.append(self._neighbours[1][0])
            self._sorted_neighbours.append(self._neighbours[0][1])
            self._sorted_neighbours.append(self._neighbours[1][1])
        else:
            self._sorted_neighbours.append(self._neighbours[1][1])
            self._sorted_neighbours.append(self._neighbours[0][1])
            self._sorted_neighbours.append(self._neighbours[1][0])

    def get_neighbour_right(self, intersection):
        """The neighbour to the right of self seen from the direction of the input intersection"""
        if not self._sorted_neighbours:
            raise StateError("Function get_neighbour_right called before sorted_neighbours were initialized")
        index = self._sorted_neighbours.index(intersection)
        new_index = (index-1)%4
        return self._sorted_neighbours[new_index]
    
    def get_neighbour_left(self, intersection):
        """The neighbour to the left of self seen from the direction of the input intersection"""
        if not self._sorted_neighbours:
            raise StateError("Function get_neighbour_right called before sorted_neighbours were initialized")
        index = self._sorted_neighbours.index(intersection)
        new_index = (index+1)%4
        return self._sorted_neighbours[new_index]

    def get_sorted_neighbours(self):
        """Return the sorted list of the neighbours. It is sorted based on the geometrical position"""
        return self._sorted_neighbours

    def add_face(self, face):
        """Add a face that this intersection is part of"""
        self._faces.append(face)

    def number_of_faces(self):
        """The known faces that this intersection is part of. Should never
        be more than four"""
        return len(self._faces)

    def get_faces(self):
        """Return a list of the known faces the intersection is part of"""
        return self._faces

    def plot(self, axis, color='red'):
        """Plot the intersection"""
        axis.plot([self._coordinates[0]], [self._coordinates[1]], [self._coordinates[2]], 'o', color=color)

def find_intersections(circle_1, circle_2):
    """Finds the two intersections between circle_1 and circle_2"""
    quat_1 = circle_1.get_rot()
    quat_2 = circle_2.get_rot()
    #which should be first
    relative_rot_2_to_1 = rotations.quaternion_multiply(quat_1, rotations.quaternion_inverse(quat_2))
    mat = rotations.quaternion_to_matrix(rotations.quaternion_inverse(relative_rot_2_to_1))

    sol = pylab.zeros(3)
    sol[1] = mat[2, 0]/pylab.sqrt(mat[2, 0]**2 + mat[2, 1]**2)
    sol[0] = -sol[1]*mat[2, 1]/mat[2, 0]
    intersection_frame_2 = sol
    
    matrix_2 = rotations.quaternion_to_matrix(quat_2) #should probably be 2
    intersection_1 = Intersection(pylab.squeeze(pylab.array(matrix_2*pylab.transpose(\
        pylab.matrix(intersection_frame_2)))))
    intersection_2 = Intersection(-intersection_1.get_coordinates())

    mat = rotations.quaternion_to_matrix(rotations.quaternion_inverse(quat_1))

    position_1_1 = math.atan2(*pylab.squeeze(pylab.array(mat*pylab.transpose(\
        pylab.matrix(intersection_1.get_coordinates()))))[:2])
    position_1_2 = math.atan2(*pylab.squeeze(pylab.array(mat*pylab.transpose(\
        pylab.matrix(intersection_2.get_coordinates()))))[:2])
    
    mat = rotations.quaternion_to_matrix(rotations.quaternion_inverse(quat_2))
    position_2_1 = math.atan2(*pylab.squeeze(pylab.array(mat*pylab.transpose(\
        pylab.matrix(intersection_1.get_coordinates()))))[:2])
    position_2_2 = math.atan2(*pylab.squeeze(pylab.array(mat*pylab.transpose(\
        pylab.matrix(intersection_2.get_coordinates()))))[:2])

    circle_1.add_intersection(intersection_1, position_1_1)
    circle_1.add_intersection(intersection_2, position_1_2)
    circle_2.add_intersection(intersection_1, position_2_1)
    circle_2.add_intersection(intersection_2, position_2_2)
    
    return intersection_1, intersection_2

def find_all_intersections(circles):
    """Given a list of Circles returns a list of all Intersections"""
    intersections = []
    for circle_1, circle_2 in itertools.combinations(circles, 2):
        this_intersection_1, this_intersection_2 = find_intersections(circle_1, circle_2)
        intersections.append(this_intersection_1)
        intersections.append(this_intersection_2)
    return intersections


class Path(object):
    """A path is an ordered set of Intersections"""
    def __init__(self, node_array = None):
        if node_array:
            self.nodes = node_array
        else:
            self.nodes = []

    def add_node(self, node):
        """Add an Intersection"""
        self.nodes.append(node)

    def add_nodes(self, nodes):
        """Add a list of Intersections"""
        self.nodes += nodes

    def get_node(self, index):
        """Return the index-th Intersection of the Path"""
        try:
            return self.nodes[index]
        except IndexError:
            return None

    def get_nodes(self):
        """Return all Intersections"""
        return self.nodes

    def head(self):
        """Return the Intersection at the end of the Path"""
        return self.nodes[-1]

    def length(self):
        """The number of Intersections in the path"""
        return len(self.nodes)

    def inverse(self):
        """Return a copy of this path but with the order reversed"""
        new_path = Path()
        new_path.nodes = self.nodes[::-1]
        return new_path

    def equal(self, path):
        """Check if this Path contains the same Intersections as another Path"""
        return sorted(path.nodes) == sorted(self.nodes)
    
    def copy(self):
        """Return a copy of this Path"""
        new_path = Path()
        new_path.nodes = self.nodes[:]
        return new_path

    def __add__(self, path):
        new_path = Path()
        new_path.nodes = self.nodes[:] + path.nodes[:]
        return new_path

    def remove_double(self):
        """If the last Intersection is the same as the first, the last is removed"""
        if self.nodes[0] == self.nodes[-1]:
            self.nodes = self.nodes[:-1]

    def order_independent_hash(self):
        """Hash calculated from the nodes contained"""
        sorted_nodes = sorted(self.nodes)
        return hash(str(sorted_nodes))

    def get_gap_size(self):
        """Find the point furthest away from the edges of this path and return the distance.
        It projects the path on its two principal components before."""

        class PcaProjector(object):
            """Project points on the sets principal components"""
            def __init__(self, coordinates):
                self._coordinates = coordinates
                self._coordinates_average = pylab.average(coordinates, axis=0)
                coordinates_translated = self._coordinates - self._coordinates_average
                #self._left_unitary, singular_values, right_unitary = pylab.svd(coordinates_translated.T)
                self._left_unitary, _, _ = pylab.svd(coordinates_translated.T)
                self._coordinates_rotated = pylab.dot(self._left_unitary.T, self._coordinates.T)

            def coordinates(self):
                """The input coordinates"""
                return self._coordinates

            def projected(self, dim):
                """The coordinates projected on the first dim principal components
                given in the pc coordinate system"""
                return self._coordinates_rotated[:dim]

            def back_transform(self, points):
                """Transform the points from the pc system to the original system"""
                return pylab.dot(self._left_unitary, points) + self._coordinates_average
                
        def get_center_from_3_lines(lines):
            """Gets the point inside these three lines that are furthest away from them"""
            equation_matrix = pylab.matrix([[lines[0, 1, 1] - lines[0, 0, 1],
                                             lines[0, 0, 0] - lines[0, 1, 0],
                                             -pylab.sqrt((lines[0, 1, 0] - lines[0, 0, 0])**2 +
                                                         (lines[0, 1, 1] - lines[0, 0, 1])**2)],
                                            [lines[1, 1, 1] - lines[1, 0, 1],
                                             lines[1, 0, 0] - lines[1, 1, 0],
                                             -pylab.sqrt((lines[1, 1, 0] - lines[1, 0, 0])**2 +
                                                         (lines[1, 1, 1] - lines[1, 0, 1])**2)],
                                            [lines[2, 1, 1] - lines[2, 0, 1],
                                             lines[2, 0, 0] - lines[2, 1, 0],
                                             -pylab.sqrt((lines[2, 1, 0] - lines[2, 0, 0])**2 +
                                                         (lines[2, 1, 1] - lines[2, 0, 1])**2)]])
            equation_vector = pylab.matrix([[lines[0, 0, 0] * (lines[0, 1, 1] - lines[0, 0, 1]) -
                                             lines[0, 0, 1] * (lines[0, 1, 0] - lines[0, 0, 0])],
                                            [lines[1, 0, 0] * (lines[1, 1, 1] - lines[1, 0, 1]) -
                                             lines[1, 0, 1] * (lines[1, 1, 0] - lines[1, 0, 0])],
                                            [lines[2, 0, 0] * (lines[2, 1, 1] - lines[2, 0, 1]) -
                                             lines[2, 0, 1] * (lines[2, 1, 0] - lines[2, 0, 0])]])

            solution = pylab.solve(equation_matrix, equation_vector)
            solution = pylab.squeeze(pylab.array(solution))
            center = solution[:2]
            distance = solution[2]

            # project center on the lines
            projected_centers = []
            projected_centers.append((lines[0, 0] + pylab.dot((center - lines[0, 0]),
                                                              (lines[0, 1] - lines[0, 0])) /
                                      pylab.norm(lines[0, 1] - lines[0, 0])**2 * (lines[0, 1] - lines[0, 0])))
            projected_centers.append((lines[1, 0] + pylab.dot((center - lines[1, 0]),
                                                              (lines[1, 1] - lines[1, 0])) /
                                      pylab.norm(lines[1, 1] - lines[1, 0])**2 * (lines[1, 1] - lines[1, 0])))
            projected_centers.append((lines[2, 0] + pylab.dot((center - lines[2, 0]),
                                                              (lines[2, 1] - lines[2, 0])) /
                                      pylab.norm(lines[2, 1] - lines[2, 0])**2 * (lines[2, 1] - lines[2, 0])))

            angles = []
            angles.append(math.atan2(*(projected_centers[0]-center)[::-1]))
            angles.append(math.atan2(*(projected_centers[1]-center)[::-1]))
            angles.append(math.atan2(*(projected_centers[2]-center)[::-1]))

            def mod_dist(angle_1, angle_2):
                """The smallest distance between two angles"""
                return min(abs(angle_1-angle_2), abs(angle_1-angle_2+2.*pi), abs(angle_1-angle_2-2.*pi))

            if (mod_dist(angles[0], angles[1]) + mod_dist(angles[1], angles[2]) +
                mod_dist(angles[2], angles[0]) > (2.*pi - 0.001)):
                is_closed = True
            else:
                is_closed = False

            #Get angle of vector from center to projected center (using atan2)

            #if there is a gap of pi it is open
            #test this by summing the distances between the angles. == 2 pi -> closed < 2 pi -> open
            
            return center, distance, is_closed

        pca_projector = PcaProjector(pylab.array([n.get_coordinates() for n in self.nodes]))
        projected = pca_projector.projected(2)

        projected_permuted = projected.copy()
        projected_permuted[:, :-1] = projected[:, 1:]
        projected_permuted[:, -1] = projected[:, 0]
        all_lines = pylab.array(zip(list(projected.T), list(projected_permuted.T)))
        #lines = array([[list(point0) for point0 in line0]  for line0 in lines])
        all_lines = [l for l in all_lines]

        centers_distances_closed = []
        for line_1, line_2, line_3 in itertools.combinations(all_lines, 3):
            centers_distances_closed.append(get_center_from_3_lines(pylab.array([line_1, line_2, line_3])))

        centers_distances = [i for i in centers_distances_closed if i[2]]
        best_c_and_d = min(centers_distances, key=lambda x: abs(x[1]))

        # transform center back to the original coordinate system
        center = pca_projector.back_transform(pylab.array(list(best_c_and_d[0]) + [0.]))
        
        #ax.plot(best_c_and_d[0][0], best_c_and_d[0][1], 'o', color='red')
        #limiting_lines = array(best_c_and_d[3])

        return best_c_and_d[1], center

    def plot(self, axis, color='black'):
        """Plot the path"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        verts = pylab.array([tuple(node.get_coordinates()) for node in self.nodes])
        poly = Poly3DCollection([verts])
        poly.set_color(color)
        poly.set_edgecolor('black')
        axis.add_collection3d(poly)

def path_sanity_check(path):
    """check that neighbours in path are true neighbours"""
    for i in range(path.length()):
        if not (path.nodes[i] in path.nodes[(i+1)%path.length()].get_all_neighbours()):
            raise AssertionError("path has no-neighbour connection")
        if not (path.nodes[i] in path.nodes[(i+path.length()-1)%path.length()].get_all_neighbours()):
            raise AssertionError("path has no-neighbour connection")

def find_region(start_intersection):
    """Find and return all the regions that contain the start_intersection and that have not
    been found before"""
    regions = []
    for second_intersection in start_intersection.get_sorted_neighbours():
        new_path = Path([start_intersection, second_intersection])
        intersection = second_intersection.get_neighbour_right(start_intersection)
        while intersection != start_intersection:
            new_path.add_node(intersection)
            intersection = new_path.head().get_neighbour_right(new_path.get_node(-2))
        #if not new_path.nodes in [face.nodes for face in start_intersection.get_faces()]:
        if not new_path.order_independent_hash() in [face.order_independent_hash() for face
                                                     in start_intersection.get_faces()]:
            regions.append(new_path)
            for node in new_path.get_nodes():
                node.add_face(new_path)
    return regions

            
def simulate(number_of_images):
    """Simulate number_of_images random orientations and find the point furthest
    away from any pattern, return this distance."""
    pylab.seed()
    circles = [Circle(rotations.random_quaternion()) for _ in range(number_of_images)]

    intersections = find_all_intersections(circles)

    for circle_0 in circles:
        circle_0.sort_intersections()

    for circle_0 in circles:
        circle_0.set_intersection_neighbours()

    for intersection_0 in intersections:
        intersection_0.sort_neighbours()

    for intersection_0 in intersections:
        assert len(intersection_0.get_neighbours()) == 2, \
               "intersection has %d directions. Should be 2" % len(intersection_0.get_neighbours())

    regions = []
    for index in range(len(intersections)):
        if intersections[index].number_of_faces() < 4:
            regions += find_region(intersections[index])

    for intersection_0 in intersections:
        assert intersection_0.number_of_faces() == 4, \
               "Intersection %d is part of %d faces" % (intersections.index(intersection_0),
                                                        intersection_0.number_of_faces())
        
    largest_region = max(regions, key=lambda x: x.get_gap_size()[0])
    return largest_region.get_gap_size()[0]

def random_color():
    """A random valid color"""
    import matplotlib
    return matplotlib.colors.cnames.keys()[pylab.random_integers(100)]

def plot_many(circles=None, intersections=None, faces=None):
    """Plot circles, intersections and faces."""
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    #from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(1)
    fig.clear()
    axis = fig.add_subplot(111, projection='3d')

    if circles != None:
        for circle_0 in circles:
            circle_0.plot(axis, 'blue')
    if intersections != None:
        for intersection_0 in intersections:
            intersection_0.plot(axis, 'red')
    if faces != None:
        for face_0 in faces:
            face_0.plot(axis, random_color())

    pylab.show()



# for c in circles:
#     c.plot()
# for r in regions:
#     r.plot(matplotlib.colors.cnames.keys()[random_integers(100)])
    
# ax.set_xlim((-1, 1))
# ax.set_ylim((-1, 1))
# ax.set_zlim((-1, 1))
# draw()

class Gaps(object):
    """Container for the largest gaps of models simulated"""
    def __init__(self, gaps, number_of_images, doc=None):
        self._gaps = gaps
        if doc:
            self._doc = doc
        else:
            self._doc = "No doc provided"
        self._number_of_images = number_of_images

    def gaps(self):
        """Returns the table of gaps. First index is for the different
        image counts. Second index is the repetitions."""
        return self._gaps

    def doc(self):
        """Returns information on the data"""
        return self._doc

    def number_of_images(self):
        """Returns a list of all the image counts used"""
        return self._number_of_images

def calculate_gaps(number_of_images_list, number_of_repetitions, filename):
    """Pickle the result to file"""
    watch = time_tools.StopWatch()
    gap_average = []
    gap_std = []
    gap_median = []
    gaps_all = []
    for number_of_images in number_of_images_list:
        #gaps = []
        watch.start()
        jobs = ((number_of_images,),)*number_of_repetitions
        gaps = parallel.run_parallel(jobs, simulate, quiet=True)
        # for i in range(number_of_repetiotions):
        #     gaps.append(gap_gap_from_N(N))
        #     print "gap_gap = %g" % gaps[-1]
        gap_average.append(pylab.average(gaps))
        gap_std.append(pylab.std(gaps))
        gap_median.append(pylab.median(gaps))
        gaps_all.append(gaps)
        watch.stop()
        print "(number_of_images = %d) average = %g, std = %g (time: %s)" % \
              (number_of_images, gap_average[-1], gap_std[-1], watch.str())
        
        gap_out = Gaps(pylab.array(gaps_all), number_of_images_list)
        file_handle = open(filename, 'wb')
        pickle.dump(gap_out, file_handle)
        file_handle.close()

    #return gap_average, gap_std, gap_median, array(gaps_all)

def plot_sphere():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    NUMBER_OF_IMAGES = 40
    LIM = 0.6
    pylab.seed()
    circles = [Circle(rotations.random_quaternion()) for _ in range(NUMBER_OF_IMAGES)]
    intersections = find_all_intersections(circles)
    fig_1 = pylab.figure(1)
    fig_1.clear()
    fig_1.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    axis_1 = fig_1.add_subplot(111, projection='3d')
    axis_1.set_aspect('equal')
    [circle_0.plot(axis_1, 'blue') for circle_0 in circles]
    [intersection_0.plot(axis_1, 'red') for intersection_0 in intersections]
    axis_1.set_xlim((-LIM, LIM))
    axis_1.set_ylim((-LIM, LIM))
    axis_1.set_zlim((-LIM, LIM))
    axis_1.set_axis_off()

    for circle_0 in circles:
        circle_0.sort_intersections()

    for circle_0 in circles:
        circle_0.set_intersection_neighbours()

    for intersection_0 in intersections:
        intersection_0.sort_neighbours()

    regions = []
    for index in range(len(intersections)):
        if intersections[index].number_of_faces() < 4:
            regions += find_region(intersections[index])

    fig_2 = pylab.figure(2)
    fig_2.clear()
    fig_2.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    axis_2 = fig_2.add_subplot(111, projection='3d')
    axis_2.set_aspect('equal')
    #[region_0.plot(axis_2, random_color()) for region_0 in regions if pylab.rand() < 0.25]
    [region_0.plot(axis_2, random_color()) for region_0 in regions]

    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    #verts = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]
    verts = [intersection_0.get_coordinates() for intersection_0 in intersections]
    # line = Line3DCollection([verts])
    # line.set_color('red')
    # #line.set_linestyle
    # #line.set_edgecolor('black')
    # line.set_linewidth(4)
    # lines = axis_2.add_collection3d(line)

    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # small_number = 0.001
    # for p in verts:
    #     poly = Poly3DCollection([[p*(1.+small_number)+small_number*pylab.rand(3),
    #                               p*(1.+small_number)+small_number*pylab.rand(3),
    #                               p*(1.+small_number)+small_number*pylab.rand(3)]])
    #     poly.set_edgecolor('black')
    #     #poly.set_color('red')
    #     poly.set_facecolor('red')
    #     poly.set_linewidth(5)
    #     axis_2.add_collection3d(poly)


    # # for p in verts:
    # #     axis_2.plot([p[0]], [p[1]], [p[2]]f, 'o', color='blue')
    
    # # axis_2.plot([1,1,1,1,-1,-1,-1,-1],
    # #             [1,1,-1,-1,1,1,-1,-1],
    # #             [1,-1,1,-1,1,-1,1,-1],'o')

    axis_2.set_xlim((-LIM, LIM))
    axis_2.set_ylim((-LIM, LIM))
    axis_2.set_zlim((-LIM, LIM))
    axis_2.set_axis_off()

def plot_sphere_vtk():
    import vtk_polygon_plotter as vpl
    NUMBER_OF_IMAGES = 10
    pylab.seed()
    circles = [Circle(rotations.random_quaternion()) for _ in range(NUMBER_OF_IMAGES)]
    intersections = find_all_intersections(circles)
    for circle_0 in circles: circle_0.sort_intersections()
    for circle_0 in circles: circle_0.set_intersection_neighbours()
    for intersection_0 in intersections: intersection_0.sort_neighbours()
    regions = []
    for index in range(len(intersections)):
        if intersections[index].number_of_faces() < 4:
            regions += find_region(intersections[index])

    vpl.plot_polygons([[node.get_coordinates() for node in r.nodes] for r in regions])


def plot_sphere_mayavi():
    import mayavi_polygon_plotter as mpl
    NUMBER_OF_IMAGES = 10
    pylab.seed()
    circles = [Circle(rotations.random_quaternion()) for _ in range(NUMBER_OF_IMAGES)]
    intersections = find_all_intersections(circles)
    for circle_0 in circles: circle_0.sort_intersections()
    for circle_0 in circles: circle_0.set_intersection_neighbours()
    for intersection_0 in intersections: intersection_0.sort_neighbours()
    regions = []
    for index in range(len(intersections)):
        if intersections[index].number_of_faces() < 4:
            regions += find_region(intersections[index])

    mpl.plot_polygons([[node.get_coordinates() for node in r.nodes] for r in regions])


if __name__ == "__main__a":
    plot_sphere_mayavi()
