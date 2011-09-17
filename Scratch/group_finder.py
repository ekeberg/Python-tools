"""
Program that uses the diffusion map to visualize relations in a group of people
"""
import pylab

class Relations:
    """Stores the mutual relations between people in a group"""
    def __init__(self, people):
        self._people = {}
        for i, person in enumerate(people):
            self._people[person] = i
        self._matrix = pylab.zeros((len(self._people),)*2)
        self._matrix += pylab.diag(10.0*pylab.ones(len(self._people)))

    def add_relation(self, person1, person2, weight = 1.0):
        """Add a one-way relation. Person1 likes Person2"""
        self._matrix[self._people[person1], self._people[person2]] = weight

    def add_mutual_relation(self, person1, person2, weight = 1.0):
        """Add a two-way relation. Person1 likes Person2 and Person2 likes Person1"""
        self._matrix[self._people[person1], self._people[person2]] = weight
        self._matrix[self._people[person2], self._people[person1]] = weight

    def add_relations(self, person1, persons):
        """Person 1 likes everyone in the list persons"""
        for person2 in persons:
            self.add_relation(person1, person2)

    def get_matrix(self):
        """Return a matrix of the relations that can be used for analysis"""
        return self._matrix

    def get_number_of_people(self):
        """Number of people (and size of the matrix returned by get_matrix()"""
        return len(self._people)

    def get_name(self, index):
        """Name corresponding to the person at a specific
        index in the matris returned by get_matrix()"""
        return [item[0] for item in self._people.items() if item[1] == index][0]
    
    def get_relation(self, person1, person2):
        """Check if Person1 likes Person2"""
        if self._matrix[self._people[person1], self._people[person2]]:
            return 1.0
        else:
            return 0.0

class DiffusionMap():
    """
    Class used to calculate the diffusion coordinates for a
    set of datapoints given a similarity_matrix"""
    def __init__(self):
        self._alpha = 0.0
        self._output_length = 3
        self._matrix = None
        self._normalized_matrix = None
        self._low_d_space = None

    def set_alpha(self, alpha):
        """
        The alpha parameter.
        alpha = 1 : Makes the density of point equal everywhere to
                    handle non-uniform sampling
        alpha = 0 : Only do a standard normalization of the matrix
        """
        self._alpha = alpha
    def get_alpha(self):
        """Return the alpha parameter"""
        return self._alpha
    def set_output_length(self, length):
        """The dimensionality of the resulting diffusion coordinates"""
        self._output_length = length
    def get_output_length(self):
        """The dimensionality of the resulting diffusion coordinates"""
        return self._output_length
    def set_similarity_matrix(self, matrix):
        """This matrix contains ths similarities between the objects.
        and is used for analyzation"""
        self._matrix = matrix
    def get_similarity_matrix(self):
        """This matrix contains ths similarities between the objects.
        and is used for analyzation"""
        return self._matrix

    def _normalize_distance_matrix(self):
        """Normalize with alpha and a regular normalization"""
        column_sum = sum(self._matrix, axis=0)
        distance_matrix = (pylab.matrix(pylab.diag(column_sum**-self._alpha))*
                           pylab.matrix(self._matrix)*
                           pylab.matrix(pylab.diag(column_sum**-self._alpha)))
        column_sum = pylab.squeeze(pylab.array(pylab.sum(distance_matrix, axis=0)))
        self._normalized_matrix = pylab.matrix(pylab.diag(1.0/column_sum))*pylab.matrix(distance_matrix)

    def calculate_diffusion_coordinates(self):
        """calculate and return diffusion coordinates"""
        self._normalize_distance_matrix()
        graph_matrix = pylab.transpose(self._normalized_matrix)

        eigenvalues_unsorted, eigenvectors_unsorted = pylab.eigh(graph_matrix)
        eig_zip = zip(eigenvalues_unsorted, range(len(eigenvalues_unsorted)))
        eig_zip.sort()
        eig_zip.reverse()
        eigenvalue_order = pylab.int32(pylab.array(eig_zip)[:, 1])
        eigenvalues = eigenvalues_unsorted[eigenvalue_order]
        #eigenvectors = [eigenvectors_unsorted[i] for
        eigenvectors = eigenvectors_unsorted[:, eigenvalue_order]
        scaled_eigenvectors = eigenvectors*pylab.diag(eigenvalues)
        #scaled_eigenvectors = diag(eigenvalues)*eigenvectors

        self._low_d_space = pylab.zeros((pylab.shape(self._matrix)[0], self.get_output_length()))
        for image_index in range(pylab.shape(self._matrix)[0]):
            for dim in range(self.get_output_length()):
                self._low_d_space[image_index, dim] = scaled_eigenvectors[image_index, dim+1]
        
        return self._low_d_space


def plot_scout_relations():
    """Plot relationships between the scouts at Spejarna"""
    relations_test = Relations(['foo1', 'foo2', 'foo3', 'foo4', 'foo5'])
    relations_test.add_relation('foo1', 'foo2')
    relations_test.add_relation('foo1', 'foo3')
    relations_test.add_relation('foo2', 'foo1')
    relations_test.add_relation('foo2', 'foo3')
    relations_test.add_relation('foo3', 'foo1')
    relations_test.add_relation('foo3', 'foo2')
    relations_test.add_relation('foo3', 'foo4')
    relations_test.add_relation('foo4', 'foo5')
    relations_test.add_relation('foo5', 'foo2')

    scouter = Relations(['ingrid', 'andrew', 'johannes', 'sofia', 'anna',
                         'wille', 'alva', 'wilma', 'annie', 'daniel w',
                         'vera', 'johanna', 'malin', 'axel', 'alex',
                         'emil', 'jacob', 'patrik', 'erik s', 'eddie',
                         'daniel f', 'daniel o', 'erik f', 'gustav'])
    scouter.add_relations('ingrid', ['wille', 'andrew', 'daniel w'])
    scouter.add_relations('andrew', ['axel', 'ingrid', 'wille', 'annie', 'alva', 'emil']) #emil
    scouter.add_relations('johannes', ['alva', 'ingrid', 'daniel w'])
    scouter.add_relations('sofia', ['wilma', 'anna', 'alva'])
    scouter.add_relations('anna', ['alva', 'wilma', 'sofia'])
    scouter.add_relations('wille', ['ingrid', 'andrew', 'daniel w', 'alex', 'emil']) #emil
    scouter.add_relations('alva', ['sofia', 'anna', 'wilma'])
    scouter.add_relations('wilma', ['anna', 'alva', 'sofia', 'wille'])
    scouter.add_relations('annie', ['ingrid', 'wille', 'andrew', 'axel', 'alva', 'anna'])
    scouter.add_relations('daniel w', ['alex'])
    scouter.add_relations('vera', ['johanna', 'malin'])
    scouter.add_relations('malin', ['daniel w', 'ingrid'])
    scouter.add_relations('axel', ['andrew', 'ingrid', 'wille', 'emil']) #emil

    scouter.add_relations('emil', ['wille', 'ingrid', 'andrew', 'axel', 'alva', 'daniel w'])
    scouter.add_relations('jacob', ['erik f', 'erik s', 'gustav'])
    scouter.add_relations('patrik', ['emil'])
    scouter.add_relations('erik s', ['erik f', 'gustav', 'jacob'])
    scouter.add_relations('eddie', ['patrik', 'daniel o', 'daniel f', 'erik s'])
    scouter.add_relations('daniel f', ['daniel o', 'erik f', 'erik s', 'eddie', 'gustav'])
    scouter.add_relations('daniel o', ['daniel f', 'eddie', 'erik f'])
    scouter.add_relations('erik f', ['gustav', 'jacob', 'erik s', 'daniel o', 'daniel f'])


    # scouter.add_mutual_relation('andrew', 'ingrid', -1.0)
    # scouter.add_mutual_relation('alva', 'wilma', -1.0)
    # scouter.add_mutual_relation('alva', 'anna', -1.0)
    # scouter.add_mutual_relation('sofia', 'wilma', -1.0)
    # scouter.add_mutual_relation('sofia', 'anna', -1.0)

    relations = scouter

    distance_matrix = 0.5*relations.get_matrix() + 1.0*pylab.transpose(relations.get_matrix()) + 0.5

    diffusion_map_calculator = DiffusionMap()
    diffusion_map_calculator.set_alpha(0.0)
    diffusion_map_calculator.set_output_length(3)
    diffusion_map_calculator.set_similarity_matrix(distance_matrix)
    low_d_space = diffusion_map_calculator.calculate_diffusion_coordinates()

    fig = pylab.figure(1)
    fig.clf()
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(low_d_space[:, 1], low_d_space[:, 0],'o')
    for i, coordinate in enumerate(low_d_space):
        ax.text(coordinate[1], coordinate[0], relations.get_name(i))

    for i in range(pylab.shape(distance_matrix)[0]):
        for j in range(pylab.shape(distance_matrix)[1]):
            if relations.get_relation(relations.get_name(i), relations.get_name(j)):
                ax.arrow(low_d_space[i, 1] + 0.1*(low_d_space[j, 1] - low_d_space[i, 1]),
                         low_d_space[i, 0] + 0.1*(low_d_space[j, 0] - low_d_space[i, 0]),
                         0.8*(low_d_space[j, 1] - low_d_space[i, 1]),
                         0.8*(low_d_space[j, 0] - low_d_space[i, 0]))

    pylab.show()

plot_scout_relations()
