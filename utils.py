import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt


class DirectedAngleInterval():
    def __init__(self, angle1, angle2, direction):
        # direction = 'ccw' or 'cw'
        # angle1 and angle2 are in range(-180, 180], or both None, in this case, the interval is a full circle
        # from angle1 to angle2 in the given direction
        # if direction == 'ccw', then angle1 < angle2
        # if direction == 'cw', then angle1 > angle2
        if angle1 is None:
            assert angle2 is None
            if direction == 'ccw':
                angle1, angle2 = -180, 180
            elif direction == 'cw':
                angle1, angle2 = 180, -180
        else:
            assert angle1 > -180 and angle1 <= 180
            assert angle2 > -180 and angle2 <= 180
        
        
        self.angle1 = angle1
        self.angle2 = angle2
        self.direction = direction

    @property
    def angle_length(self):
        if self.direction == 'ccw':
            if self.angle1 > 0 and self.angle2 < 0:
                return (180 - self.angle1) + (180 + self.angle2)
            else:
                return self.angle2 - self.angle1
        elif self.direction == 'cw':
            if self.angle1 < 0 and self.angle2 > 0:
                return (180 + self.angle1) + (180 - self.angle2)
            else:
                return self.angle1 - self.angle2

    def contain(self, direction):
        if self.direction == 'ccw':
            if self.angle1 > 0 and self.angle2 < 0:
                if direction > self.angle1 or direction < self.angle2:
                    return True
                else:
                    return False
            else:
                if self.angle1 < direction < self.angle2:
                    return True
                else:
                    return False
        elif self.direction == 'cw':
            raise NotImplemented
        
    def get_mid_angle(self):
        if self.direction == 'ccw':
            return (self.angle1 + self.angle2) / 2
        elif self.direction == 'cw':
            raise NotImplemented


    def max(self, target_direction):
        # target_direction in range(-180, 180], since np.arctan2 returns (-180, 180]
        # return the angle in this interval that follows the target_direction best
        assert target_direction > -180 and target_direction <= 180
        if self.contain(target_direction):
            return target_direction
        else:
            mid_angle = self.get_mid_angle()
            if DirectedAngleInterval(self.angle2, round_angle(mid_angle + 180), 'ccw').contain(target_direction):
                return self.angle2
            else:
                return self.angle1
    
    def min(self, target_direction):
        # target_direction in range[-180, 180)
        # return the angle in this interval that follows the target_direction worst
        return round_angle(self.max(round_angle(target_direction+180)))
        


def round_angle(angle):
    # # round angle to (-180, 180]
    if angle > 180:
        angle -= 360
    elif angle <= -180:
        angle += 360
    assert angle > -180 and angle <= 180
    return angle

def find_first_right(prev_direction, next_directions):
    next_direction_val = [_[0] for _ in next_directions]

    if prev_direction > min(next_direction_val):
        order = np.argsort(next_direction_val)
        next_direction_val = np.array(next_direction_val)[order]

        # find the first element in next_directions that is smaller than prev_direction
        return order[np.where(np.array(next_direction_val) < prev_direction)[0][-1]]
    else:
        # find the largest element in next_directions
        return np.argmax(next_direction_val)
    

def angle_linspace(start, end, clockwise, resolution):
    # start in range(-180, 180)
    # end in range(-180, 180)
    # clockwise: bool, True if we want to go clockwise, False if we want to go counterclockwise
    # resolution: float, the distance between two points on the arc

    # TODO rewrite this function with DirectedAngleInterval class
    if start is None:
        assert end is None
        return np.arange(180, -180, -resolution) # a full circle, travel clockwise, and the region is to the right of the arc
    elif start < end <= 0 and not clockwise:
        return np.arange(start, end, resolution)
    elif end < start <= 0 and clockwise:
        return np.arange(start, end, -resolution)
    elif 0 <= start < end and not clockwise:
        return np.arange(start, end, resolution)
    elif start < 0 and end > 0 and not clockwise:
        return np.arange(start, end, resolution)
    elif start > end >= 0 and clockwise:
        return np.arange(start, end, -resolution)
    elif start < 0 and end > 0 and clockwise:
        return np.concatenate([np.arange(start, -180, -resolution), np.arange(180, end, -resolution)])
    elif start > 0 and end < 0 and not clockwise:
        return np.concatenate([np.arange(start, 180, resolution), np.arange(-180, end, resolution)])
    elif start > 0 and end < 0 and clockwise:
        return np.arange(start, end, -resolution)
    elif start == end:
        return np.array([start])
    else:
        raise NotImplemented

class Region():
    def __init__(self):
        pass
        self.arc_ls = []
        self.relation_ls = {} # relationship with all the circles in the figure, key: circle_id
        self.p1_to_p2_ls = []
        # self.right_or_left_ls = [] # 没啥用

    def add_arc(self, arc, in_out: str, p1_to_p2: bool):
        # in_out: the region is inside the circle or outside the circle
        # in_out: 'in' or 'out'
        # p1_to_p2: the direction of the arc that we travel

        self.arc_ls.append(arc)
        self.relation_ls[arc.circle.id] = in_out + '_boundary'
        # this region is in (or out) the circle of this arc, and the arc is the boundary of the region

        self.p1_to_p2_ls.append(p1_to_p2)

        # if (in_out == 'in' and p1_to_p2) or (in_out == 'out' and not p1_to_p2):
        #     self.right_or_left_ls.append('left')
        # else:
        #     self.right_or_left_ls.append('right')

    def checkIfClockwise(self):
        # compute how many degrees we turned right and how many degrees we turned left as we travel the region
        # positive for right, negative for left
        degree = 0
        if len(self.arc_ls) == 1:
            # this region has only one arc (a full circle)
            self.clockwise = True
            return
        
        for i in range(len(self.arc_ls)):
            arc = self.arc_ls[i]
            p1_to_p2 = self.p1_to_p2_ls[i]
            if p1_to_p2: # the angle when traveling along the arc
                # turn left
                degree -= arc.angle_length
            else:
                # turn right
                degree += arc.angle_length

            # the angle when moving from the end point of the arc to the start point of the next arc
            if i != len(self.arc_ls) - 1:
                prev_end_direction = arc.dir_1_to_2 if p1_to_p2 else arc.dir_2_to_1
                next_start_direction = round_angle(self.arc_ls[i+1].dir_2_to_1+180) if self.p1_to_p2_ls[i+1] else round_angle(self.arc_ls[i+1].dir_1_to_2+180)
                degree += DirectedAngleInterval(prev_end_direction, next_start_direction, 'cw').angle_length
            else:
                # from end point of the last arc to start point of the first arc
                prev_end_direction = arc.dir_1_to_2 if p1_to_p2 else arc.dir_2_to_1
                next_start_direction = round_angle(self.arc_ls[0].dir_2_to_1+180) if self.p1_to_p2_ls[0] else round_angle(self.arc_ls[0].dir_1_to_2+180)
                degree += DirectedAngleInterval(prev_end_direction, next_start_direction, 'cw').angle_length

        if degree > 0:
            self.clockwise = True
        else:
            self.clockwise = False
        

    def finalize(self, circle_ls):
        """
        Should be called when all arcs are added (start point == end point).
        In this function we check if the region is closed, and if there is full circle inside of the region.
        """
        # check if the region is closed
        assert self.is_closed()

        self.checkIfClockwise()

        # check and save the relationship of this region with other circles
        # ignore the circles of the arcs of this region
        ignore_circle_id_ls = [arc.circle_id for arc in self.arc_ls]
        for circle in circle_ls:
            if circle.id in ignore_circle_id_ls:
                pass
            else:
                # the given circle is not the circle of any arc in this region
                # thus its boundary is either inside or outside of this region
                # we simply pick a point on the circle and check its relationship with this region
                self.relation_ls[circle.id] = self.checkRelationWithPoint((circle.x, circle.y+circle.r))
        # sort the relation_ls by circle_id
        self.relation_ls = {k: v for k, v in sorted(self.relation_ls.items(), key=lambda item: item[0])}
       
       # if there is circle totally inside of this region
        # if 'in' in self.relation_ls.values():
        #     a = 1
            
    def checkRelationWithPoint(self, point):
        # if the point is exactly on the boundary of the region, then we consider it is outside of the region

        # https://en.wikipedia.org/wiki/Point_in_polygon
        # implement Ray Casting Algorithm first
        direction = 0 # the ray starts from point and goes to the right
        n_cross = 0
        for arc in self.arc_ls:
            # arc.checkRelationWithRay(point, direction), we only use direction = 0 for simplicity, thus no need to implement this function
            # compute the intersection of the line of the ray and the circle of the arc
            if point[1] > arc.circle.y + arc.circle.r or point[1] < arc.circle.y - arc.circle.r:
                # the ray does not intersect with the circle of the arc
                # thus the ray does not intersect with the arc
                n_cross += 0
            else:
                if point[1] == arc.circle.y + arc.circle.r:
                    # the ray intersects with the circle of the arc at the top point
                    # circle_line_intersect = [(arc.circle.x, arc.circle.y + arc.circle.r)]
                    n_cross += 0 # if the point is exactly on the boundary of the region, then we consider it is outside of the region
                elif point[1] == arc.circle.y - arc.circle.r:
                    # the ray intersects with the circle of the arc at the bottom point
                    # circle_line_intersect = [(arc.circle.x, arc.circle.y - arc.circle.r)]
                    n_cross += 0 # if the point is exactly on the boundary of the region, then we consider it is outside of the region
                else:
                    # the ray intersects with the circle of the arc at two points
                    # compute the intersection points
                    # https://stackoverflow.com/a/1084899/13114834
                    dx = np.sqrt(arc.circle.r**2 - (point[1] - arc.circle.y)**2)
                    if arc.circle.x - dx >= point[0]:
                        n_cross += int(arc.checkRelationWithPointOnCircle(arc.circle.x - dx, point[1]))
                    if arc.circle.x + dx >= point[0]:
                        n_cross += int(arc.checkRelationWithPointOnCircle(arc.circle.x + dx, point[1]))

        if self.clockwise:
            return 'in' if n_cross % 2 == 1 else 'out'
        else:
            return 'out' if n_cross % 2 == 1 else 'in'
            





    def distance_to_point(self, x, y):
        # return max_distance, min_distance
        # max_distance: the max distance from (x, y) to all arcs in this region
        # min_distance: the min distance from (x, y) to all arcs in this region
        max_distance = -np.inf
        min_distance = np.inf
        for arc in self.arc_ls:
            max_distance_arc, min_distance_arc = arc.distance_to_point(x, y)
            max_distance = max(max_distance, max_distance_arc)
            min_distance = min(min_distance, min_distance_arc)
        return max_distance, min_distance



    def is_closed(self):
        start_point_id = self.arc_ls[0].point_1_id if self.p1_to_p2_ls[0] else self.arc_ls[0].point_2_id
        end_point_id = self.arc_ls[-1].point_2_id if self.p1_to_p2_ls[-1] else self.arc_ls[-1].point_1_id
        return start_point_id == end_point_id


    
    @property
    def range(self):
        assert self.is_closed()
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for arc in self.arc_ls:
            arc_range = arc.range
            x_min = min(x_min, arc_range['x_min'])
            x_max = max(x_max, arc_range['x_max'])
            y_min = min(y_min, arc_range['y_min'])
            y_max = max(y_max, arc_range['y_max'])
        return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        

    def plot(self, axes, circle_ls, resolution=0.1, alpha=0.3, color='g', debug=False, adapt_range=False):        
        # for each arc, generate list of points on the arc
        all_x = []
        all_y = []
        for i in range(len(self.arc_ls)):
            arc = self.arc_ls[i]            
            theta1, theta2 = arc.get_angle()

            from_theta = theta1 if self.p1_to_p2_ls[i] else theta2
            to_theta = theta2 if self.p1_to_p2_ls[i] else theta1

            # if self.p1_to_p2_ls[i], then we are in counterclockwise direction
            # else we are in clockwise direction
            theta_ls = angle_linspace(from_theta, to_theta, clockwise=not self.p1_to_p2_ls[i], resolution=resolution)
                            
            x = arc.circle.x + arc.circle.r * np.cos(theta_ls * np.pi / 180)
            y = arc.circle.y + arc.circle.r * np.sin(theta_ls * np.pi / 180)
            # plt.figure()
            # if debug:
            #     plt.scatter(x, y, c=np.arange(len(x)))
            all_x.append(x)
            all_y.append(y)

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        extra = 0.3


        if 'in' in self.relation_ls.values():
            if self.clockwise:
                # we travel along the arcs in a clockwise direction
                # and the region is to the right hand side of our travel direction
                # the region is inside the boundary of the arcs
                # we need to exclude the circles that are totally inside of this region
                # we sample points on the circles that is totally inside of this region

                # find the circles that are totally inside of this region
                circle_id_ls = [i for i in self.relation_ls.keys() if self.relation_ls[i] == 'in']
                if len(circle_id_ls) == 1:
                    # circle_ls[circle_id_ls[0]]
                    x, y, r = circle_ls[circle_id_ls[0]].x, circle_ls[circle_id_ls[0]].y, circle_ls[circle_id_ls[0]].r
                    theta_ls = np.arange(-180, 180, resolution)
                    x_ls = x + r * np.cos(theta_ls * np.pi / 180)
                    y_ls = y + r * np.sin(theta_ls * np.pi / 180)
                    all_x = np.concatenate([all_x, x_ls])
                    all_y = np.concatenate([all_y, y_ls])
                else:
                    raise NotImplemented
            else:
                # TODO
                # raise NotImplemented
                pass


        fill_right(x=all_x, y=all_y, axes=axes, alpha=alpha, color=color, debug=debug, extra=extra)

        if adapt_range:
            region_range = self.range
            x_len = region_range['x_max'] - region_range['x_min']
            y_len = region_range['y_max'] - region_range['y_min']
            axes.set_xlim(region_range['x_min'] - x_len * extra, region_range['x_max'] + x_len * extra)
            axes.set_ylim(region_range['y_min'] - y_len * extra, region_range['y_max'] + y_len * extra)



class Arc():
    def __init__(self, circle, arc_id, point_1, point_2, circle_ls):
        # TODO use DirectedAngleInterval to store theta1 and theta2 directly
        # this arc from point_1 to point_2 in counterclockwise direction
        # we need circle_ls to identify the relationship of this arc with other circles (either in or out)
        self.circle_id = circle.id
        self.circle = circle # the circle object, the information is not updated, only use its (x,y,r)
        self.arc_id = arc_id

        if point_1 is None:
            self.point_1_id = None
            self.point_2_id = None
        else:
            self.point_1_id = point_1.id
            self.point_2_id = point_2.id
        
        self.point_1 = point_1
        self.point_2 = point_2

        self.theta1, self.theta2 = self.get_angle()

        self.n_used = 0

        # identify the relationship of this arc with other circles (either in or out)
        # self.check_in_out(circle_ls)

        self.get_tangent_direction()

    @property
    def angle_length(self):
        # from point_1 to point_2 in counterclockwise direction
        # how many degrees we need to turn
        return DirectedAngleInterval(self.theta1, self.theta2, 'ccw').angle_length

        raise NotImplemented

    def distance_to_point(self, x, y):
        # return max_distance, min_distance
        # max_distance: the max distance from (x, y) to this arc
        # min_distance: the min distance from (x, y) to this arc
        target_direction = np.arctan2(self.circle.y - y, self.circle.x - x) * 180 / np.pi
        farthest_p, closest_p = self.get_farthest_point(target_direction)
        max_distance = d(x, y, farthest_p.x, farthest_p.y)
        min_distance = d(x, y, closest_p.x, closest_p.y)
        return max_distance, min_distance
    
    def checkRelationWithPointOnCircle(self, x, y):
        # check if a point (x, y) that on the circle of this arc, lies on the arc
        # return True if the point lies on the arc, False otherwise
        assert self.circle.checkPointRelation(x, y) == 'on'
        angle = self.getAngleForPointOnCircle(x, y)
        return DirectedAngleInterval(self.theta1, self.theta2, 'ccw').contain(angle)
        

    def checkCircleRelation(self, circle):
        # identify the relationship of this arc with other circles (either in or out)
        if circle.id == self.circle_id:
            return 'is'
        else:
            if circle in self.circle.is_inside_circle_ls:
                # self.circle is inside of the given circle
                # thus this arc is inside of the given circle
                return 'in'
            elif circle in self.circle.is_outside_circle_ls:
                # self.circle is outside of the given circle
                # thus this arc is outside of the given circle
                return 'out'
            else:
                target_direction = np.arctan2(self.circle.y - circle.y, self.circle.x - circle.x) * 180 / np.pi
                farthest_p, closest_p = self.get_farthest_point(target_direction)
                # farthest_p 是 arc上距离circle圆心最远的点
                # closest_p 是 arc上距离circle圆心最近的点
                if circle.checkPointRelation(farthest_p.x, farthest_p.y):
                    assert circle.checkPointRelation(closest_p.x, closest_p.y)
                    return 'in'
                else:
                    assert not circle.checkPointRelation(closest_p.x, closest_p.y)
                    return 'out'




    # def check_in_out(self, circle_ls):
    #     # TODO used in Region.finalize(), only requires check if this arc is inside or outside of circles with no intersection points
    #     # identify the relationship of this arc with other circles (either in or out)
    #     self.in_out_ls = []
    #     for circle in circle_ls:
    #         if circle.id == self.circle_id:
    #             self.in_out_ls.append('is')
    #         else:
    #             if circle in self.circle.is_inside_circle_ls:
    #                 # self.circle is inside of the given circle
    #                 # thus this arc is inside of the given circle
    #                 self.in_out_ls.append('in')
    #             elif circle in self.circle.is_outside_circle_ls:
    #                 # self.circle is outside of the given circle
    #                 # thus this arc is outside of the given circle
    #                 self.in_out_ls.append('out')
    #             else:
    #                 target_direction = np.arctan2(self.circle.y - circle.y, self.circle.x - circle.x) * 180 / np.pi
    #                 farthest_p, closest_p = self.get_farthest_point(target_direction)
                    # farthest_p 是 arc上距离circle圆心最远的点
                    # closest_p 是 arc上距离circle圆心最近的点
                    # if circle.checkPointRelation(farthest_p.x, farthest_p.y):
                    #     assert circle.checkPointRelation(closest_p.x, closest_p.y)
                    #     self.in_out_ls.append('in')
                    # else:
                    #     assert not circle.checkPointRelation(closest_p.x, closest_p.y)
                    #     self.in_out_ls.append('out')

                    # # more computation required
                    # # the closet point on self.circle to the center of the given circle
                    # closest_angle = np.arctan2(circle.y - self.circle.y, circle.x - self.circle.x) * 180 / np.pi
                    # farest_angle = round_angle(closest_angle + 180)

                    # if closest_angle < self.theta1 < self.theta2 < farest_angle:
                    #     # point_1 is closest to the center of the given circle
                    #     # point_2 is farest to the center of the given circle
                    #     # check if point_1 is inside of the given circle
                    #     point_1_in_circle = circle.checkPointRelation(self.point_1.x, self.point_1.y)
                    #     point_2_in_circle = circle.checkPointRelation(self.point_2.x, self.point_2.y)
                    #     if point_1_in_circle and point_2_in_circle:
                    #         self.in_out_ls.append('in')
                    #     else:
                    #         raise NotImplemented
                    # elif self.theta1 < self.theta2 < farest_angle < 0 < closest_angle:
                    #     # 1 is closest, 2 is farest
                    #     point_1_in_circle = circle.checkPointRelation(self.point_1.x, self.point_1.y)
                    #     point_2_in_circle = circle.checkPointRelation(self.point_2.x, self.point_2.y)
                    #     if point_1_in_circle and point_2_in_circle:
                    #         self.in_out_ls.append('in')
                    #     elif not point_1_in_circle and not point_2_in_circle:
                    #         self.in_out_ls.append('out')
                    #     else:
                    #         raise NotImplemented
                    # elif farest_angle < self.theta1 < self.theta2 < 0 < closest_angle:
                    #     # 2 is closest, 1 is farest
                    #     point_1_in_circle = circle.checkPointRelation(self.point_1.x, self.point_1.y)
                    #     point_2_in_circle = circle.checkPointRelation(self.point_2.x, self.point_2.y)
                    #     if point_1_in_circle and point_2_in_circle:
                    #         self.in_out_ls.append('in')
                    #     elif not point_1_in_circle and not point_2_in_circle:
                            # self.in_out_ls.append('out')
                        # else:
                            # raise NotImplemented
                    # elif farest_angle == self.theta1 < self.theta2 < 0 < closest_angle:
                    #     # 2 is closest, 1 is farest
                    #     point_1_in_circle = circle.checkPointRelation(self.point_1.x, self.point_1.y)
                    #     point_2_in_circle = circle.checkPointRelation(self.point_2.x, self.point_2.y)
                    #     if point_1_in_circle and point_2_in_circle:
                    #         self.in_out_ls.append('in')
                    #     elif not point_1_in_circle and not point_2_in_circle:
                    #         self.in_out_ls.append('out')
                    #     else:
                    #         raise NotImplemented
                    # else:
                        # raise NotImplemented


    def get_tangent_direction(self):
        if self.point_1 is not None:
            self.dir_1_to_2 = round_angle(180 - np.arctan2(self.point_2.x - self.circle.x, self.point_2.y - self.circle.y) * 180 / np.pi)
            self.dir_2_to_1 = round_angle(- np.arctan2(self.point_1.x - self.circle.x, self.point_1.y - self.circle.y) * 180 / np.pi)
        else:
            pass


    def getAngleForPointOnCircle(self, x, y):
        assert self.circle.checkPointRelation(x, y) == 'on'
        # return the angle of the point on the circle of this arc
        # the angle is in range(-180, 180]
        return round_angle(np.arctan2(y - self.circle.y, x - self.circle.x) * 180 / np.pi)
        


    def get_angle(self):
        # TODO reimplement this function with getAngleForPointOnCircle
        # get theta_1 and theta_2, in counterclockwise direction
        # theta_1 is the smaller angle
        # [0, 360) ?
        if self.point_1_id is None:
            return None, None
        else:
            theta_1 = round_angle(np.arctan2(self.point_1.y - self.circle.y, self.point_1.x - self.circle.x) * 180 / np.pi)
            theta_2 = round_angle(np.arctan2(self.point_2.y - self.circle.y, self.point_2.x - self.circle.x) * 180 / np.pi)
            return theta_1, theta_2

    def get_other_point(self, point_id):
        assert point_id in [self.point_1_id, self.point_2_id]
        if point_id == self.point_1_id:
            return self.point_2_id
        else:
            return self.point_1_id
        
    def get_plt_patch(self, linewidth):
        if self.theta1 is not None:
            arc = patches.Arc(xy=(self.circle.x, self.circle.y),
                            width=self.circle.r*2,
                            height=self.circle.r*2,
                            angle=0, # rotate angle
                            theta1=self.theta1,
                            theta2=self.theta2,
                            color='r',
                            linewidth=linewidth)
        else:
            # this arc is a full circle
            arc = patches.Circle(xy=(self.circle.x, self.circle.y), radius=self.circle.r, color='r', fill=False, linewidth=linewidth)
        return arc
        
    def plot(self, axes, linewidth=2):
        # plot the arc on the figure
        # if axes is a list
        if isinstance(axes, list):
            for ax in axes:
                arc = self.get_plt_patch(linewidth)
                ax.add_patch(arc)
        else:
            arc = self.get_plt_patch(linewidth)
            axes.add_patch(arc)

        # also see https://stackoverflow.com/a/45579263/13114834 to draw arrow on arc

    # a function get the farthest point and closest point on the arc to a given direction
    def get_farthest_point(self, direction):
        # direction in range(-180, 180)
        # return farthest point, closest point
        arc_angle_interval = DirectedAngleInterval(self.theta1, self.theta2, 'ccw')
        max_angle = arc_angle_interval.max(direction)
        min_angle = arc_angle_interval.min(direction)
        farthest_point = Point(x=self.circle.x + self.circle.r * np.cos(max_angle * np.pi / 180), y=self.circle.y + self.circle.r * np.sin(max_angle * np.pi / 180), id=None, circle_1=None, circle_2=None, angles=None)
        closest_point = Point(x=self.circle.x + self.circle.r * np.cos(min_angle * np.pi / 180), y=self.circle.y + self.circle.r * np.sin(min_angle * np.pi / 180), id=None, circle_1=None, circle_2=None, angles=None)
        return farthest_point, closest_point

    @property
    def range(self):
        x_max_p, x_min_p = self.get_farthest_point(0)
        y_max_p, y_min_p = self.get_farthest_point(90)
        return {'x_max': x_max_p.x, 'x_min': x_min_p.x, 'y_max': y_max_p.y, 'y_min': y_min_p.y}


class Circle():
    def __init__(self, x, y, r, id):
        self.x = x
        self.y = y
        self.r = r
        self.id = id
        self.intersection_point_ls = []
        self.intersection_point_angle_ls = []
        self.is_inside_circle_ls = [] # list of other circles that this circle is inside of
        self.is_outside_circle_ls = [] # list of other circles that this circle is outside of

    def add_intersection_point(self, point):
        self.intersection_point_ls.append(point)
        angle = np.arctan2(point.y - self.y, point.x - self.x)
        self.intersection_point_angle_ls.append(angle)

    def find_arcs(self, arc_id, circle_ls):
        # return a list of Arc objects

        if len(self.intersection_point_ls) == 0:
            # if no intersection point, return a full circle
            arc_ls = [Arc(circle=self, arc_id=arc_id, point_1=None, point_2=None, circle_ls=circle_ls)]
            n_arc = 1
        else:
            # sort the intersection points by angle
            order = np.argsort(self.intersection_point_angle_ls)
            self.intersection_point_ls = np.array(self.intersection_point_ls)[order]
            self.intersection_point_angle_ls = np.array(self.intersection_point_angle_ls)[order]

            # find the arcs
            arc_ls = []
            n_arc = 0
            for i in range(len(self.intersection_point_ls) - 1):
                arc = Arc(circle=self, arc_id=arc_id+n_arc, point_1=self.intersection_point_ls[i], point_2=self.intersection_point_ls[i + 1], circle_ls=circle_ls)
                arc_ls.append(arc)
                n_arc += 1
            arc = Arc(circle=self, arc_id=arc_id+n_arc, point_1=self.intersection_point_ls[-1], point_2=self.intersection_point_ls[0], circle_ls=circle_ls)
            arc_ls.append(arc)
            n_arc = len(arc_ls)

        return arc_ls, len(arc_ls)
        
    def plot(self, axes, color):
        circle = patches.Circle(xy=(self.x, self.y), radius=self.r, color=color, fill=False, label='C{}'.format(self.id))
        axes.add_patch(circle)

    def checkPointRelation(self, point_x, point_y):
        """
        Determines the relationship of a given point with respect to the circle.

        The function returns:
        - 'inside' if the point is inside the circle,
        - 'on' if the point is on the circle, and
        - 'outside' if the point is outside the circle.

        :param point_x: The x-coordinate of the point to check.
        :param point_y: The y-coordinate of the point to check.
        :return: A string indicating the relationship ('inside', 'on', or 'outside').
        """
        distance_squared = (point_x - self.x) ** 2 + (point_y - self.y) ** 2
        radius_squared = self.r ** 2

        if np.isclose(distance_squared, radius_squared):
            return 'on'
        elif distance_squared < radius_squared:
            return 'inside'
        else:
            return 'outside'
    
    def set_inside_circle(self, circle):
        self.is_inside_circle_ls.append(circle)

    def set_outside_circle(self, circle):
        self.is_outside_circle_ls.append(circle)

class Point():
    def __init__(self, x, y, id, circle_1, circle_2, angles):
        self.x = x
        self.y = y
        self.id = id
        self.connected_arc_ids = []
        self.circle_1 = circle_1
        self.circle_2 = circle_2
        self.angles = angles

        self.axes_plotted = {}

    def set_id(self, id):
        assert self.id is None
        self.id = id

    def add_arc(self, arc_id):
        self.connected_arc_ids.append(arc_id)
    
    def plot(self, axes, textsize, alpha):
        # scatter on the given axes
        if isinstance(axes, list):
            for ax in axes:
                if ax not in self.axes_plotted:
                    ax.scatter(self.x, self.y, s=20, label='P{}'.format(self.id))
                    ax.text(self.x, self.y, 'P{}'.format(self.id), fontsize=textsize, alpha=alpha)
                    self.axes_plotted[ax] = True
        else:
            if axes not in self.axes_plotted:
                axes.scatter(self.x, self.y, s=20, label='P{}'.format(self.id))
                axes.text(self.x, self.y, 'P{}'.format(self.id), fontsize=textsize, alpha=alpha)
                self.axes_plotted[axes] = True

def d(p1x, p1y, p2x, p2y):
    return np.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)


def get_direction(of, at, towards):
    # get direction of "circle_of" at "point_at, towards both inside and outside of "circle_towards"
    # return a tuple of (inside, outside)

    direction_1 = - np.arctan2(of.x - at[0], of.y - at[1]) * 180 / np.pi
    assert direction_1 >= -180 and direction_1 < 180

    direction_2 = (direction_1 - 180)
    if direction_2 >= 180:
        direction_2 -= 360
    elif direction_2 < -180:
        direction_2 += 360
    assert direction_2 >= -180 and direction_2 < 180

    # direction_out: vector(from c_towards to at) * direction_out > 0
    if np.dot(np.array([at[0]-towards.x, at[1]-towards.y]), np.array([np.cos(direction_1 * np.pi / 180), np.sin(direction_1 * np.pi / 180)])) > 0:
        direction_out = direction_1
        direction_in = direction_2
        
    else:
        direction_out = direction_2
        direction_in = direction_1

    return (direction_in, direction_out)



    # direction_2 = direction_1 - 180
    # assert direction_2 >= -180 and direction_2 < 180

    # which direction is inside? depends on start from center of circle_towards then go to center of circle_of, then go to point_at
    # if the angle is increasing, then the direction is inside
    # if the angle is decreasing, then the direction is outside

    # get the angle from center of circle_towards to center of circle_of

    

    # raise NotImplemented


def intersect_two_circle(c1, c2):
    # return two intersection points
    # if no intersection, return None
    c1x, c1y, r1 = c1.x, c1.y, c1.r
    c2x, c2y, r2 = c2.x, c2.y, c2.r
    if d(c1x, c1y, c2x, c2y) > r1 + r2:
        # save c1 in c2 or c2 in c1 for later use
        c1.set_outside_circle(c2)
        c2.set_outside_circle(c1)
        return None
    elif d(c1x, c1y, c2x, c2y) < abs(r1 - r2):
        if r1 < r2: # c1 is inside c2
            c1.set_inside_circle(c2)
            c2.set_outside_circle(c1)
        elif r1 > r2: # c2 is inside c1
            c2.set_inside_circle(c1)
            c1.set_outside_circle(c2)
        else:
            raise NotImplemented
        return None
    else:
        a = (r1**2 - r2**2 + d(c1x, c1y, c2x, c2y)**2) / (2 * d(c1x, c1y, c2x, c2y))
        h = np.sqrt(r1**2 - a**2)
        x2 = c1x + a * (c2x - c1x) / d(c1x, c1y, c2x, c2y)
        y2 = c1y + a * (c2y - c1y) / d(c1x, c1y, c2x, c2y)
        p1x = x2 + h * (c2y - c1y) / d(c1x, c1y, c2x, c2y)
        p1y = y2 - h * (c2x - c1x) / d(c1x, c1y, c2x, c2y)
        p2x = x2 - h * (c2y - c1y) / d(c1x, c1y, c2x, c2y)
        p2y = y2 + h * (c2x - c1x) / d(c1x, c1y, c2x, c2y)

        p1 = (p1x, p1y)
        p2 = (p2x, p2y)

        d_c1_in_c2_at_p1, d_c1_out_c2_at_p1 = get_direction(of=c1, at=p1, towards=c2)
        d_c1_in_c2_at_p2, d_c1_out_c2_at_p2 = get_direction(of=c1, at=p2, towards=c2)

        d_c2_in_c1_at_p1, d_c2_out_c1_at_p1 = get_direction(of=c2, at=p1, towards=c1)
        d_c2_in_c1_at_p2, d_c2_out_c1_at_p2 = get_direction(of=c2, at=p2, towards=c1)

        return {'p1': Point(p1x, p1y, None, c1, c2, {'d_c1_in_c2': d_c1_in_c2_at_p1, 'd_c1_out_c2': d_c1_out_c2_at_p1, 'd_c2_in_c1': d_c2_in_c1_at_p1, 'd_c2_out_c1': d_c2_out_c1_at_p1}),
                'p2': Point(p2x, p2y, None, c1, c2, {'d_c1_in_c2': d_c1_in_c2_at_p2, 'd_c1_out_c2': d_c1_out_c2_at_p2, 'd_c2_in_c1': d_c2_in_c1_at_p2, 'd_c2_out_c1': d_c2_out_c1_at_p2})}
    
def plot_two_circle(c1x, c1y, c2x, c2y, r1, r2, label: list = None):
    circle = plt.Circle((c1x, c1y), r1, color='b', fill=False, label=label[0])
    plt.gca().add_artist(circle)
    circle = plt.Circle((c2x, c2y), r2, color='g', fill=False, label=label[1])
    plt.gca().add_artist(circle)

    xlim = plt.gca().get_xlim()
    xlim_min = min(xlim[0], c1x - r1, c2x - r2)
    xlim_max = max(xlim[1], c1x + r1, c2x + r2)
    plt.xlim((xlim_min, xlim_max))

    ylim = plt.gca().get_ylim()
    ylim_min = min(ylim[0], c1y - r1, c2y - r2)
    ylim_max = max(ylim[1], c1y + r1, c2y + r2)
    plt.ylim((ylim_min, ylim_max))

    plt.gca().set_aspect('equal', adjustable='box')

def fill_right(x, y, axes, alpha, color='lightgreen', debug=False, extra=0.3):
    """
    Fill the region to the right of a given polygon.

    Parameters:
    x (list): X-coordinates of the polygon vertices.
    y (list): Y-coordinates of the polygon vertices.
    color (str): Color to fill the polygon. Default is light green.
    debug (bool): If True, show additional plot details for debugging.
    extra (float): Relative extra margin to add to the encompassing rectangle.

    The function fills inside the polygon if the vertices are in clockwise order,
    and fills outside if they are in counterclockwise order.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of elements.")

    if len(x) < 3:
        # exit the function
        Warning("At least 3 points are required to form a polygon.")
        return

    # Calculate the signed area of the polygon
    n = len(x)
    signed_area = 0.5 * sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))
    orientation = "Counterclockwise" if signed_area > 0 else "Clockwise"

    # Create the encompassing rectangle dynamically based on x and y range
    margin_x, margin_y = (max(x) - min(x)) * extra, (max(y) - min(y)) * extra
    x_min, x_max, y_min, y_max = min(x) - margin_x, max(x) + margin_x, min(y) - margin_y, max(y) + margin_y
    x_rect = [x_min, x_max, x_max, x_min, x_min]
    y_rect = [y_min, y_min, y_max, y_max, y_min]

    if debug:
        axes.scatter(x, y, c=np.arange(len(x)))
        axes.set_title('From blue to yellow')
        
    if orientation == "Counterclockwise":
        # Fill outside for counterclockwise orientation
        x = x[::-1]  # Reverse the points
        y = y[::-1]
        x_combined = np.concatenate([x_rect, x])
        y_combined = np.concatenate([y_rect, y])
        axes.fill(x_combined, y_combined, color=color, alpha=alpha)
    else:
        # Fill inside for clockwise orientation
        axes.fill(x, y, color=color, alpha=alpha)




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #### test intersect_two_circle
    if False:
        def test_intersect_two_circle(c1x, c1y, c2x, c2y, r1, r2, axes):
            c1 = Circle(c1x, c1y, r1, 1)
            c2 = Circle(c2x, c2y, r2, 2)

            c1.plot(axes, color='C1')
            c2.plot(axes, color='C2')

            intersects = intersect_two_circle(c1, c2)
            p1, p2 = intersects['p1'], intersects['p2']
            p1.set_id(1)
            p2.set_id(2)
            # Arc(circle, arc_id, point_1, point_2)
            a1 = Arc(c1, 1, p1, p2)

            plt.gca().set_aspect('equal', adjustable='box')
            
            p1.plot(axes, textsize=10, alpha=1)
            p2.plot(axes, textsize=10, alpha=1)
            plt.legend()
            
            # a1.plot(axes)

            # # Direction of C1 at P1 towards inside C2
            # # plot a the tangent of C1 at P1 (the line perpendicular to the line connecting C1 and P1)
            plt.plot([p1.x, p1.x + 10 * np.cos(p1.angles['d_c1_in_c2'] * np.pi / 180)], [p1.y, p1.y + 10 * np.sin(p1.angles['d_c1_in_c2'] * np.pi / 180)],
                     color='C1', label='Direction of C1 at P1 towards inside C2')
            
            # # Direction of C1 at P1 towards outside C2
            # # plot a the tangent of C1 at P1 (the line perpendicular to the line connecting C1 and P1)
            plt.plot([p1.x, p1.x + 10 * np.cos(p1.angles['d_c1_out_c2'] * np.pi / 180)], [p1.y, p1.y + 10 * np.sin(p1.angles['d_c1_out_c2'] * np.pi / 180)],
                     color='C1', linestyle='--', label='Direction of C1 at P1 towards outside C2')
            
            # Direction of C2 at P1 towards inside C1
            # plot a the tangent of C2 at P1 (the line perpendicular to the line connecting C2 and P1)
            plt.plot([p1.x, p1.x + 10 * np.cos(p1.angles['d_c2_in_c1'] * np.pi / 180)], [p1.y, p1.y + 10 * np.sin(p1.angles['d_c2_in_c1'] * np.pi / 180)],
                     color='C2', label='Direction of C2 at P1 towards inside C1')
            
            # # Direction of C2 at P1 towards outside C1
            # # plot a the tangent of C2 at P1 (the line perpendicular to the line connecting C2 and P1)
            plt.plot([p1.x, p1.x + 10 * np.cos(p1.angles['d_c2_out_c1'] * np.pi / 180)], [p1.y, p1.y + 10 * np.sin(p1.angles['d_c2_out_c1'] * np.pi / 180)],
                     color='C2', linestyle='--', label='Direction of C2 at P1 towards outside C1')

            # legend outside of the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()


            plt.title("""Direction of C1 at P1 towards inside C2: {:.2f}\n
    Direction of C1 at P1 towards outside C2: {:.2f}\n
    Direction of C2 at P1 towards inside C1: {:.2f}\n
    Direction of C2 at P1 towards outside C1: {:.2f}""".format(p1.angles['d_c1_in_c2'], p1.angles['d_c1_out_c2'], p1.angles['d_c2_in_c1'], p1.angles['d_c2_out_c1']))
            

        fig = plt.figure()
        ax = fig.add_subplot(111)
        r1 = 30
        c2x = np.random.randint(-30, 30)
        c2y = np.random.randint(-30, 30)
        r2_min = max(0, d(0, 0, c2x, c2y) - r1)
        r2_max = d(0, 0, c2x, c2y) + r1
        r2 = np.random.randint(r2_min, r2_max)

        test_intersect_two_circle(c1x = 0, c1y = 0, c2x = c2x, c2y = c2y, r1 = r1, r2 = r2, axes = ax)


        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(121)
        # test_intersect_two_circle(c1x = 0, c1y = 0, c2x = 7, c2y = 0, r1 = 8, r2 = 2, axes = ax)

        # ax = fig.add_subplot(122)
        # test_intersect_two_circle(c1x = 0, c1y = 0, c2x = 12, c2y = 0, r1 = 10, r2 = 5, axes = ax)

        plt.show()

    #### test np.arctan2
    if False:
        theta_ls = np.linspace(-np.pi, np.pi, 100)
        x_ls = np.cos(theta_ls)
        y_ls = np.sin(theta_ls)
        arctan2_ls = np.arctan2(y_ls, x_ls)
        plt.scatter(x_ls, y_ls, c=arctan2_ls)
        plt.colorbar()
        plt.title('Value of np.arctan2(y, x)')
        plt.show()

    ### test fill_right
    num_points = 100
    radius = 1
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    # counter clockwise
    fill_right(x_circle, y_circle)
    # clockwise
    fill_right(x_circle[::-1], y_circle[::-1])
    