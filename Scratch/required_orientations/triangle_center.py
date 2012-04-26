from pylab import *
import itertools

fig = figure(1)
fig.clf()
ax = fig.add_subplot(111)

#corners = random((3,2))

lines_disjoint = array([[[0.9, 0.1], [0.1, 0.1]],
                        [[0.1, 0.4], [0.4, 0.6]],
                        [[0.6, 0.6], [0.8, 0.2]]])
lines_joint = [[[0.9, 0.1], [0.1, 0.1]],
               [[0.1, 0.1], [0.5, 0.8]],
               [[0.5, 0.8], [0.9, 0.1]]]
lines_joint_reverse = [[[0.1, 0.1], [0.9, 0.1]],
                       [[0.5, 0.8], [0.1, 0.1]],
                       [[0.9, 0.1], [0.5, 0.8]]]
lines_open = array([[[0.9, 0.2], [0.1, 0.1]],
                        [[0.1, 0.7], [0.4, 0.6]],
                        [[0.6, 0.6], [0.8, 0.2]]])
lines_5 = [[[0.6, 0.1], [0.3, 0.1]],
           [[0.3, 0.1], [0.1, 0.4]],
           [[0.1, 0.4], [0.4, 0.8]],
           [[0.4, 0.8], [0.7, 0.4]],
           [[0.7, 0.4], [0.6, 0.1]]]
lines = lines_5
for line in lines:
    line = array(line)
    ax.plot(line[:,0], line[:,1], color='blue', lw=2)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')


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
    solution = solve(m,d)

    solution = squeeze(array(solution))
    c = solution[:2]
    d = solution[2]

    # project center on the lines

    projected_point_1 = lines[0,0] + dot((c - lines[0,0]), (lines[0,1] - lines[0,0])) / norm(lines[0,1] - lines[0,0])**2 * (lines[0,1] - lines[0,0])
    projected_point_2 = lines[1,0] + dot((c - lines[1,0]), (lines[1,1] - lines[1,0])) / norm(lines[1,1] - lines[1,0])**2 * (lines[1,1] - lines[1,0])
    projected_point_3 = lines[2,0] + dot((c - lines[2,0]), (lines[2,1] - lines[2,0])) / norm(lines[2,1] - lines[2,0])**2 * (lines[2,1] - lines[2,0])

    angle_1 = math.atan2(*(projected_point_1-c)[::-1])
    angle_2 = math.atan2(*(projected_point_2-c)[::-1])
    angle_3 = math.atan2(*(projected_point_3-c)[::-1])

    def mod_dist(a, b):
        return min(abs(a-b), abs(a-b+2.*pi), abs(a-b-2.*pi))

    print "angles = (%g, %g, %g)" % (angle_1, angle_2, angle_3)
    print "1->2: %g, 2->3: %g, 3->1: %g" % (mod_dist(angle_1, angle_2), mod_dist(angle_2, angle_3), mod_dist(angle_3, angle_1))
    if mod_dist(angle_1, angle_2) + mod_dist(angle_2, angle_3) + mod_dist(angle_3, angle_1) > (2.*pi - 0.001):
        is_closed = True
    else:
        is_closed = False

    #Get angle of vector from center to projected center (using atan2)

    #if there is a gap of pi it is open
    #test this by summing the distances between the angles. == 2 pi -> closed < 2 pi -> open
    
    return c, d, is_closed


centers_distances_closed = []
for line_1, line_2, line_3 in itertools.combinations(lines, 3):
    c, d, closed = get_center_from_3_lines(array([line_1, line_2, line_3]))
    centers_distances_closed.append((c, d, closed, (line_1, line_2, line_3)))
    print closed
    if closed:
        ax.plot(c[0], c[1], 'o', color='blue')
    else:
        ax.plot(c[0], c[1], 'o', color='black')
    print "(%d, %d, %d) -> %g" % (lines.index(line_1), lines.index(line_2), lines.index(line_3), d)

centers_distances = [i for i in centers_distances_closed if i[2]]
best_c_and_d = min(centers_distances, key=lambda x: abs(x[1]))
ax.plot(best_c_and_d[0][0], best_c_and_d[0][1], 'o', color='red')
limiting_lines = array(best_c_and_d[3])
ax.plot(limiting_lines[0,:,0], limiting_lines[0,:,1], color='red', lw=2)
ax.plot(limiting_lines[1,:,0], limiting_lines[1,:,1], color='red', lw=2)
ax.plot(limiting_lines[2,:,0], limiting_lines[2,:,1], color='red', lw=2)
    
# c, d = get_center_from_3_lines(lines)
# ax.plot(c[0], c[1], 'o')
# print d

draw()
    

def get_center_from_triangle(corners):
    #for plotting we need the first element duplicated last
    plotting_corners = zeros((4,2))
    plotting_corners[:3,:] = corners; plotting_corners[-1,:] = corners[0,:]
    ax.plot(plotting_corners[:,0],plotting_corners[:,1])

    m = matrix([[corners[1,1] - corners[0,1], corners[0,0] - corners[1,0],
                 -sqrt((corners[1,0] - corners[0,0])**2 + (corners[1,1] - corners[0,1])**2)],
                [corners[2,1] - corners[1,1], corners[1,0] - corners[2,0],
                 -sqrt((corners[2,0] - corners[1,0])**2 + (corners[2,1] - corners[1,1])**2)],
                [corners[0,1] - corners[2,1], corners[2,0] - corners[0,0],
                 -sqrt((corners[0,0] - corners[2,0])**2 + (corners[0,1] - corners[2,1])**2)]])
    d = matrix([[corners[0,0] * (corners[1,1] - corners[0,1]) - corners[0,1] * (corners[1,0] - corners[0,0])],
                [corners[1,0] * (corners[2,1] - corners[1,1]) - corners[1,1] * (corners[2,0] - corners[1,0])],
                [corners[2,0] * (corners[0,1] - corners[2,1]) - corners[2,1] * (corners[0,0] - corners[2,0])]])

    c = solve(m,d)
    ax.plot(c[0], c[1], 'o')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')
    draw()
