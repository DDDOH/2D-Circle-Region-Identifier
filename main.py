# count number of region given a list of circles

# general for n circles, not our case (all circles have a common point, select three customers their three circles has only two intersection points)

import numpy as np
import matplotlib.pyplot as plt
import os
from utils import intersect_two_circle, Circle, Point, Arc, Region, round_angle, find_first_right


if not os.path.exists('.nosync'):
    os.makedirs('.nosync')


# PARAMETERS
n_circle = 20
PLOT_ARC = False
arclw = 0.5
PLOT_POINT = False
textsize = 10
textalpha = 0.5
PLOT_REGION = True

# Generate random circles
np.random.seed(0)

c_x = np.random.randint(0, 30, n_circle)
c_y = np.random.randint(0, 30, n_circle)
r = np.random.randint(1, 20, n_circle)

fig_x_range = (min(c_x - r) - 1, max(c_x + r) + 1)
fig_y_range = (min(c_y - r) - 1, max(c_y + r) + 1)



circle_ls = []
for i in range(n_circle):
    circle_ls.append(Circle(c_x[i], c_y[i], r[i], i))


# plot all circles
fig = plt.figure(figsize=(4,8))
ax1 = fig.add_subplot(311)
for circle in circle_ls:
    circle.plot(ax1, color='b')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.xlim(fig_x_range)
plt.ylim(fig_y_range)
plt.gca().set_aspect('equal', adjustable='box')

ax2 = fig.add_subplot(312)
for circle in circle_ls:
    circle.plot(ax2, color='b')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.xlim(fig_x_range)
plt.ylim(fig_y_range)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Region just found')

ax3 = fig.add_subplot(313)


plt.tight_layout()



# compute all intersection points and all arcs
point_id = 0
point_ls = []

for c1_id in range(n_circle):
    c1 = circle_ls[c1_id]
    for c2_id in range(c1_id + 1, n_circle):
        c2 = circle_ls[c2_id]
        intersects = intersect_two_circle(c1, c2)
        if intersects is not None:
            P1 = intersects['p1']
            P1.set_id(point_id)
            point_ls.append(P1)
            point_id += 1

            P2 = intersects['p2']
            P2.set_id(point_id)
            point_ls.append(P2)
            point_id += 1

            c1.add_intersection_point(P1)
            c1.add_intersection_point(P2)
            c2.add_intersection_point(P1)
            c2.add_intersection_point(P2)
        


n_point = len(point_ls)
            
# for each circle, find their arcs
# maintain a list of arcs, each arc can be used for two regions (left and right)
# recording if each arc is used for left or right region
arc_ls = []
arc_id = 0
for c in circle_ls:
    arc_ls_for_c, n_arc = c.find_arcs(arc_id, circle_ls)
    arc_ls += arc_ls_for_c
    arc_id += n_arc

# in arc, we have stored the point id
# we store the arc_id into each point
for arc in arc_ls:
    if arc.point_1_id is not None:
        point_ls[arc.point_1_id].add_arc(arc.arc_id)
        point_ls[arc.point_2_id].add_arc(arc.arc_id)
    # else this arc is indeed a circle


# start from any arc with any unused region
# (find the first arc with unused region)

# an arc can be twice, one for 1to2, one for 2to1
arc_used_1to2 = np.zeros(len(arc_ls), dtype=bool) # 0: unused, 1: used, 
arc_used_2to1 = np.zeros(len(arc_ls), dtype=bool) # 0: unused, 1: used

ax3.pcolormesh(np.stack([arc_used_1to2, arc_used_2to1]), cmap='gray')

region_ls = []

while not ((arc_used_1to2).all() and (arc_used_2to1).all()):
    # we still have arc with unused direction
    if len(region_ls) == 16: # 这个似乎有问题
        a = 1

    # initialize
    current_region = Region()
    # not all arcs are used for 1to2 (counterclockwise)

    # find a unused arc
    if not (arc_used_1to2).all():
        arc_id = np.where(arc_used_1to2 == False)[0][0]
        arc_used_1to2[arc_id] = True
        region_start_p_id = arc_ls[arc_id].point_1_id
        p_start_id = arc_ls[arc_id].point_1_id
        p_next_id = arc_ls[arc_id].point_2_id
        p1_to_p2 = True
    else:
        arc_id = np.where(arc_used_2to1 == False)[0][0]
        arc_used_2to1[arc_id] = True
        region_start_p_id = arc_ls[arc_id].point_2_id
        p_start_id = arc_ls[arc_id].point_2_id
        p_next_id = arc_ls[arc_id].point_1_id
        p1_to_p2 = False
        
    # assume we travel the region clockwise, then the region is outside of the circle of the arc
    current_region.add_arc(arc_ls[arc_id], in_out='out', p1_to_p2=p1_to_p2)


    only_one_circle = region_start_p_id == None # this region contains only one arc, which is a circle

    if PLOT_POINT:
        if not only_one_circle:
            point_ls[region_start_p_id].plot([ax1, ax2], textsize=textsize, alpha=textalpha)
            point_ls[p_next_id].plot([ax1, ax2], textsize=textsize, alpha=textalpha)
    if PLOT_ARC:
        arc_ls[arc_id].plot([ax1, ax2], linewidth=arclw)
        


    print('from P{} to P{} along A{} on C{}, p1 to p2'.format(p_start_id, p_next_id, arc_id, arc_ls[arc_id].circle_id))


    if not only_one_circle:
        # prepare for the next iteration
        p_prev_id = p_next_id
        prev_direction = arc_ls[arc_id].dir_1_to_2 if p1_to_p2 else arc_ls[arc_id].dir_2_to_1

        while region_start_p_id != p_prev_id:
            # find the next arc
            # we always turn right
            # list all directions that we can go further
            directions = []
            arc_id_ls = point_ls[p_prev_id].connected_arc_ids
            for _ in arc_id_ls:
                if _ != arc_id and arc_ls[_].circle.id != arc_ls[arc_id].circle.id: # not go back to the previous arc, and not continue on the same circle
                    # check 1to2 or 2to1 for each arc
                    if arc_ls[_].point_1_id == p_prev_id:
                        # we want to use this arc as 1to2
                        # which means the region is outside of the circle of the arc
                        directions.append((round_angle(arc_ls[_].dir_2_to_1 - 180), _, '1_to_2'))

                    else:
                        # we want to use this arc as 2to1
                        # which means the region is inside of the circle of the arc
                        directions.append((round_angle(arc_ls[_].dir_1_to_2 - 180), _, '2_to_1'))
            
            # find the first direction on our right
            which_arc_in_directions = find_first_right(prev_direction, next_directions=directions)

            arc_id = directions[which_arc_in_directions][1]
            is_1to2 = directions[which_arc_in_directions][2] == '1_to_2'


            # same for each iteration
            if is_1to2:
                arc_used_1to2[arc_id] = True
                ax3.pcolormesh(np.stack([arc_used_1to2, arc_used_2to1]), cmap='gray')
                p_start_id = arc_ls[arc_id].point_1_id
                p_next_id = arc_ls[arc_id].point_2_id
                current_region.add_arc(arc_ls[arc_id], in_out='out', p1_to_p2=True)
            else:
                arc_used_2to1[arc_id] = True
                ax3.pcolormesh(np.stack([arc_used_1to2, arc_used_2to1]), cmap='gray')
                p_start_id = arc_ls[arc_id].point_2_id
                p_next_id = arc_ls[arc_id].point_1_id
                current_region.add_arc(arc_ls[arc_id], in_out='in', p1_to_p2=False)

            if PLOT_POINT:
                point_ls[p_next_id].plot([ax1, ax2], textsize=textsize, alpha=textalpha)
            if PLOT_ARC:
                arc_ls[arc_id].plot([ax1, ax2], linewidth=arclw)
            if is_1to2:
                print('from P{} to P{} along A{} on C{}, p1 to p2'.format(p_start_id, p_next_id, arc_id, arc_ls[arc_id].circle_id))
            else:
                print('from P{} to P{} along A{} on C{}, p2 to p1'.format(p_start_id, p_next_id, arc_id, arc_ls[arc_id].circle_id))

            # prepare for the next iteration
            p_prev_id = p_next_id
            if is_1to2:
                prev_direction = arc_ls[arc_id].dir_1_to_2
            else:
                prev_direction = arc_ls[arc_id].dir_2_to_1

    # we have finished one region
    current_region.finalize(circle_ls) # check if region is closed, and check if there is isolate circle inside
    region_ls.append(current_region)

    if PLOT_REGION:
        current_region.plot(ax1, resolution=0.1, alpha=0.8, color='g', debug=False, circle_ls=circle_ls)
        current_region.plot(ax2, resolution=0.1, alpha=0.4, color='r', debug=False, circle_ls=circle_ls, adapt_range=True)
        # set xlim and ylim of ax2 to display only the current region
    
    if len(region_ls) == 55: # 这个似乎有问题
        a = 1
    # set debug point or save figure here
    print('region {} finished'.format(len(region_ls)))
    # save figure as retina quality
    plt.savefig('.nosync/region_{}.png'.format(len(region_ls)), dpi=300)

    current_region.plot(ax2, resolution=0.1, alpha=0.8, color='g', debug=False, circle_ls=circle_ls)