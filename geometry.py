import numpy as np
import math
from sympy import *
from scipy.optimize import fsolve
def genline(point1,point2): #point2为终止点
    line=[point1[0],point1[1],point2[0],point2[1]]
    return line
def genray(point1,point2):
    ray=[point2[0]-point1[0],point2[1]-point1[1]]
    return ray
def point_line_distance(point, line_point1, line_point2):  #点到直线的距离
    # 计算向量
    line_point1 = np.array(line_point1)
    line_point2 = np.array(line_point2)
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance

def is_point_on_segment(point, line):
    """
    判断点是否在线段上。

    参数:
    point (tuple): 待判断的点，格式为(x, y)。
    segment_start (tuple): 线段起点，格式为(x, y)。
    segment_end (tuple): 线段终点，格式为(x, y)。

    返回:
    bool: 如果点在线段上返回True，否则返回False。
    """
    x, y = point
    x1, y1 = line[0],line[1]
    x2, y2 = line[2],line[3]

    # 计算线段的向量
    dx = x2 - x1
    dy = y2 - y1

    # 计算点到线段起点的向量
    dx1 = x - x1
    dy1 = y - y1

    # 使用叉乘判断点是否与线段共线
    if (dx * dy1 - dy * dx1) != 0:
        return False  # 点不在线段所在的直线上

    # 点在直线上，现在检查点是否在线段的范围内
    # 投影点到线段上，计算投影点的参数t
    t = (dx1 * dx + dy1 * dy) / (dx ** 2 + dy ** 2)

    # 检查t是否在0到1之间，即点是否在线段上
    return 0 <= t <= 1
def point_point_distance(p1, p2): #两点之间的距离公式
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def line_point_pedal_point(line, p3): #点与直线的垂点
    """
    过p3作p1和p2相连直线的垂线, 计算垂足的坐标
    直线1：垂足坐标和p3连线
    直线2: p1和p2连线
    两条直线垂直, 且交点为垂足
    :param line: (x1, y1) (x2, y2)
    :param p3: (x3, y3)
    :return: 垂足坐标 (x, y)
    """
    if line[2] != line[0]:
        # ########## 根据点x1和x2计算线性方程的k, b
        k, b = np.linalg.solve([[float(line[0]), 1], [float(line[2]), 1]], [float(line[1]), float(line[3])])  # 得到k和b
        # #######原理: 垂直向量数量积为0
        x = np.divide(((line[2] - line[0]) * p3[0] + (line[3] - line[1]) * p3[1] - b * (line[3] - line[1])),
                      (line[2] - line[0] + k * (line[3] - line[1])))
        y = k * x + b

    else:  # 点p1和p2的连线垂直于x轴时
        x = line[0]
        y = p3[1]

    return [x, y]

def line_point_pedalline(line, p3):  #直线和直线外点的垂线
    point_pedal=line_point_pedal_point(line,p3)
    line_pedal=[point_pedal[0],point_pedal[1],p3[0],p3[1]]
    return line_pedal

def line_point_pedalline2(line,P):   #直线和直线内点的垂线
    N=[P[0]-(line[3]-line[1]),P[1]+(line[2]-line[0])]
    return [P[0],P[1],N[0],N[1]]




##  计算两直线的夹角
def GetCrossAngle(line1, line2):
    pi=math.pi
    [x1,y1,x2,y2]=line1
    [x3,y3,x4,y4]=line2
    arr_0 = np.array([(x2 - x1), (y2 - y1)])
    arr_1 = np.array([(x4 - x3), (y4 - y3)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))
    if cos_value>1:
        cos_value=1
    elif cos_value<-1:
        cos_value=-1

    return math.degrees(np.arccos(cos_value))


def cal_ang(point_1, point_2, point_3):  #三点夹角 无方向
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    try:
        B = math.degrees(math.acos(round((b * b - a * a - c * c) / (-2 * a * c),8)))
    except:
        B=0
    # B = math.degrees(math.acos(round((b * b - a * a - c * c) / (-2 * a * c), 8)))
    return B



def line_line_intersection(line1, line2):  #两直线交点
    a1 = line1[3] - line1[1]
    b1 = line1[0] - line1[2]
    c1 = a1 * line1[0] + b1 * line1[1]
    a2 = line2[3] - line2[1]
    b2 = line2[0] - line2[2]
    c2 = a2 * line2[0] + b2 * line2[1]
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return None  # lines are parallel
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    return [x, y]


# 计算圆 与 线段相交的点
def line_intersect_circle(circle_center,r,line):
    '''
    计算圆心与线段的交点
    :param circle_center: 圆心 （x0,y0）
    :param r: 半径
    :param line: 线段起始点（x1,y1）线段终点（x2,y2）
    :return: 线段与圆的交点
    '''
    x0, y0=circle_center
    x1=line[0]
    y1=line[1]
    x2=line[2]
    y2=line[3]
    if r == 0:
        return [[x1, y1]]
    #斜率不存在的情况
    if x1 == x2:
        inp = []
        if abs(r) >= abs(x1 - x0):
            #下方这个点
            p1 = x1, round(y0 - (r ** 2 - (x1 - x0) ** 2)**(0.5), 8)
            #上方这个点
            p2 = x1, round(y0 +(r ** 2 - (x1 - x0) ** 2)**(0.5), 8)
            if max(y1,y2)>=p2[1]:
                inp.append(p2)
            if min(y1,y2)<=p1[1]:
                inp.append(p1)
    else:
        #求直线y=kx+b的斜率及b
        k = (y1 - y2) / (x1 - x2)
        b0 = y1 - k * x1
        #直线与圆的方程化简为一元二次方程ax**2+bx+c=0
        a = k ** 2 + 1
        b = 2 * k * (b0 - y0) - 2 * x0
        c = (b0 - y0) ** 2 + x0 ** 2 - r ** 2
        #判别式判断解，初中知识
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            p1x = round((-b - delta**(0.5)) / (2 * a), 8)
            p2x = round((-b + delta**(0.5)) / (2 * a), 8)
            p1y = round(k * p1x + b0, 8)
            p2y = round(k * p2x + b0, 8)
            inp = [[p1x, p1y], [p2x, p2y]]
            inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
        else:
            inp = []
    return inp if inp != [] else [[x1, y1]]

def get_tangent_line(P, C, r):  #点与圆的切点
  """
  :param px: 圆外点P横坐标
  :param py: 圆外点P纵坐标
  :param cx: 圆心横坐标
  :param cy: 圆心纵坐标
  :param r:  圆半径
  :return:   过点P的俩条切线方程
  """
  # print(P,C,r)
  px=P[0]
  py=P[1]
  cx=C[0]
  cy=C[1]
  # 求点到圆心的距离
  distance = math.sqrt((px-cx)*(px-cx)+(py-cy)*(py-cy))
  # print('distance', distance)
  # 点p 到切点的距离
  length = math.sqrt(distance*distance-r*r)
  # print('length', length)
  if distance <= r:
    print("输入的数值不在范围内")
    return
  # 点到圆心的单位向量
  ux = (cx-px)/distance
  uy = (cy-py)/distance
  # print('ux', ux)
  # print('uy', uy)
  # 计算切线与圆心连线的夹角
  angle = math.asin(r/distance)
  # print('angle', math.degrees(angle))
  # 向正反两个方向旋转单位向量
  q1x = ux * math.cos(angle)  -  uy * math.sin(angle)
  q1y = ux * math.sin(angle)  +  uy * math.cos(angle)
  q2x = ux * math.cos(-angle) -  uy * math.sin(-angle)
  q2y = ux * math.sin(-angle) +  uy * math.cos(-angle)
  # 得到新座标y
  q1x = q1x * length + px
  q1y = q1y * length + py
  q2x = q2x * length + px
  q2y = q2y * length + py

  q1x=round(q1x,8)
  q1y = round(q1y, 8)
  q2x = round(q2x, 8)
  q2y = round(q2y, 8)
  return [q1x, q1y, q2x, q2y]

def point_tangency_bsm(P,E,a,attack):

    def equations(vars):
        x, y = vars
        m=a*math.sqrt((y-P[1])**2+(x-P[0])**2)/abs(x-P[0])
        A=1+(y-P[1])/(x-P[0])-m**2
        B=-2*E[0]-2*P[0]*((y-P[1])/(x-P[0]))**2+2*m*(m*P[0]+attack)
        C=((y-P[1])**2/((x-P[0])**2))*P[0]**2+(P[1]-E[1])**2-(m*P[0]+attack)**2
        eq1 = B**2-4*A*C
        eq2=a*(math.sqrt((y-P[1])**2+(x-P[0])**2)-attack)-math.sqrt((y-E[1])**2+(x-E[0])**2)
        return [eq1,eq2]

    W1 = fsolve(equations, [E[0]+100, E[1] + 100])
    W2 = fsolve(equations, [E[0]-100, E[1] - 100])
    return W1,W2
# print(point_tangency_bsm([-1,0],[5,5],0.8,1))

def point_bsm(P,E,ve,a,attack):
    def equations(vars):
        x, y = vars
        eq1 = E[1] - y + ve[1] / ve[0] * (x - E[0])
        eq2 = (sqrt((x - P[0]) ** 2 + (y - P[1]) ** 2) - attack) * a - sqrt((x - E[0]) ** 2 + (y - E[1]) ** 2)
        return [eq1, eq2]

    disPE=point_point_distance(P,E)

    W1 = fsolve(equations, [E[0]+ve[0],E[1]+ve[1]])
    W2 = fsolve(equations, [E[0]-ve[0],E[1]-ve[1]])
    # print(type(W1))
    # W1=np.array(W1)
    # W2=np.array(W2)
    # print("W1,W2,E=", W1, W2,E)
    if (W1[0]-E[0])*ve[0]>=0 and (W1[1]-E[1])*ve[1]>=0:  #######存在问题
        return W1
    # elif (W2[0]-E[0])*ve[0]>=0 and (W2[1]-E[1])*ve[1]>=0:
    else:
        return W2

def clockwise_angle(v1, v2):  #顺时针角度

    x1,y1 = v1
    x2,y2 = v2
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    # print("dot,det",dot,det)
    theta = np.arctan2(float(det), float(dot))
    theta = theta if theta>0 else 2*np.pi+theta
    return round(np.degrees(theta),8)

def clockwise_angle2(v1, v2):
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    # 叉乘
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1,v2)/TheNorm))
    theta=round(theta,8)
    if rho < 0:
        return - theta+360
    else:
        return theta



def point3_ccw_angle(A,B,C): #从左到右逆时针角度
    A=np.array(A)
    B = np.array(B)
    C = np.array(C)
    v1=A-B
    v2=C-B

    theta=clockwise_angle(v1, v2)
    theta=round(theta,8)
    return theta

def dir_route(A,B): #A为终止向量  B指向A
    A = np.array(A)
    B = np.array(B)
    C=A-B

    return C
def is_intersecting(line1, line2): #判断两线段是否相交
    # 获取两个线段的坐标
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    # 计算向量
    u = (x2 - x1, y2 - y1)
    v = (x4 - x3, y4 - y3)
    w = (x1 - x3, y1 - y3)

    # 计算向量积
    cross_u_v = u[0] * v[1] - u[1] * v[0]
    cross_u_w = u[0] * w[1] - u[1] * w[0]
    cross_v_w = v[0] * w[1] - v[1] * w[0]

    # 判断两线段是否相交
    if cross_u_v == 0:
        if cross_u_w != 0 or cross_v_w != 0:
            return False
        else:
            if x1 < x2:
                if x2 < x3 or x4 < x1:
                    return False
            else:
                if x1 < x3 or x4 < x2:
                    return False
            return True
    else:
        s = cross_u_w / cross_u_v
        t = cross_v_w / cross_u_v
        if 0 <= s <= 1 and 0 <= t <= 1:
            return True
        else:
            return False

def is_intersecting_rayandline(ray,line):#判断射线与线段的交点  射线ray[A,B]A为端点
    J=line_line_intersection(ray,line)
    #判断是否在射线上
    A=[ray[0],ray[1]]
    B=[ray[2],ray[3]]
    if (A[0]-B[0])*(A[0]-J[0])>0: #在射线上
        E=[line[0],line[1]]
        F=[line[2],line[3]]
        disEF=point_point_distance(E,F)
        disJE=point_point_distance(J,E)
        disJF = point_point_distance(J, F)
        if disJF<=disEF and disJE<=disEF:
            return True
        else:
            return False
    else:
        return False

def dis_ray_point(point,ray_p1,ray_p2): #点到射线的距离  射线ray[A,B]A为端点
    A = ray_p1
    B = ray_p2
    ray=[ray_p1[0],ray_p1[1],ray_p2[0],ray_p2[1]]
    # P=line_point_pedal_point(ray,point)
    # line=genline(P,point)
    #
    # if is_intersecting_rayandline(ray,line):
    #     dis=point_line_distance(point,A,B)
    # else:
    #     dis=point_point_distance(A,point)

    angle = cal_ang(point, ray_p1, ray_p2)
    if angle<90:
        dis=point_line_distance(point,A,B)
    else:
        dis=point_point_distance(A,point)

    return dis

def is_point_in_triangle(point,triangle): #判断点是不是在障碍物内


    if type(triangle) is not np.ndarray:
        A = triangle[0]
        B = triangle[1]
        C = triangle[2]
        P = point
        undirAPB = cal_ang(A, P, B)
        undirAPC = cal_ang(A, P, C)
        undirBPC = cal_ang(B, P, C)
        if undirAPB + undirBPC + undirAPC <= 359.99:
            return False
        else:
            return True
    else:
        A = triangle[0,0]
        B = triangle[1,0]
        C = triangle[2,0]
        P = [point[0],point[1]]
        undirAPB = cal_ang(A, P, B)
        undirAPC = cal_ang(A, P, C)
        undirBPC = cal_ang(B, P, C)
        if undirAPB + undirBPC + undirAPC <= 359.99:
            return False
        else:
            return True


def insec(p1, r1, p2, r2):  #找两圆的交点
    x = p1[0]
    y = p1[1]
    R = r1
    a = p2[0]
    b = p2[1]
    S = r2
    d = math.sqrt((abs(a - x)) ** 2 + (abs(b - y)) ** 2)
    if d > (R + S) or d < (abs(R - S)):
        #print("Two circles have no intersection")
        return None,None
    elif d == 0:
        #print("Two circles have same center!")
        return None,None
    else:
        A = (R ** 2 - S ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(R ** 2 - A ** 2)
        x2 = x + A * (a - x) / d
        y2 = y + A * (b - y) / d
        x3 = round(x2 - h * (b - y) / d, 8)
        y3 = round(y2 + h * (a - x) / d, 8)
        x4 = round(x2 + h * (b - y) / d, 8)
        y4 = round(y2 - h * (a - x) / d, 8)
        c1 = [x3, y3]
        c2 = [x4, y4]
        return c1, c2

def list_product(list1,list2):  #判断两线交点个数
    count = 0
    for i in range(len(list1)):  # 为了防止两个列表长度不一致
        try:
            count += list1[i] * list2[i]
        except:
            break
    return count

def is_nan(nan): #判断值是否为NAN
    return nan != nan

def line_obstacle_intersection(line,obstacle):  #找一条线段与多边形的交点
    line1 = genline(obstacle[0], obstacle[1])
    line2 = genline(obstacle[1], obstacle[2])
    line3 = genline(obstacle[2], obstacle[0])

    if is_intersecting(line, line1):
        Q = line_line_intersection(line, line1)
    elif is_intersecting(line, line2):
        Q = line_line_intersection(line, line2)
    else:
        Q = line_line_intersection(line, line3)
    return Q

def line_obstacles_intersection_judge(line,obstacle): #判断线段与多个多边形是否有交点
    if (type(obstacle).__name__ == 'list'):
        obstacle=np.array(obstacle)

    if len(obstacle.shape)==2:
        obstacleline01 = genline(obstacle[0], obstacle[1])
        obstacleline12 = genline(obstacle[1], obstacle[2])
        obstacleline20 = genline(obstacle[2], obstacle[0])
        if is_intersecting(line, obstacleline20) or is_intersecting(line,obstacleline01) or is_intersecting(line, obstacleline12):
            return True
        else:
            return False
    if len(obstacle.shape)==3:
        for obstacle_single in obstacle:
            obstacleline01 = genline(obstacle_single[0], obstacle_single[1])
            obstacleline12 = genline(obstacle_single[1], obstacle_single[2])
            obstacleline20 = genline(obstacle_single[2], obstacle_single[0])
            if is_intersecting(line, obstacleline20) or is_intersecting(line, obstacleline01) or is_intersecting(line,obstacleline12):
                return True

        return False
def line_obstacle_intersection_find(line,obstacle,return_insect_line=False):  #找一条线段与多边形的交点
    insect_point=[]
    insect_line=[]
    if (type(obstacle).__name__ == 'list'):
        obstacle=np.array(obstacle)

    if len(obstacle.shape)==2:
        for p in range(len(obstacle)):
            if p>=len(obstacle)-1:
                obstacleline = genline(obstacle[p], obstacle[0])
                another_point=obstacle[0]
            else:
                obstacleline = genline(obstacle[p], obstacle[p+1])
                another_point=obstacle[p+1]
            if is_intersecting(line, obstacleline) and is_point_on_segment(another_point, line) == False:
                insect_point.append(line_line_intersection(line, obstacleline))
                insect_line.append(obstacleline)
        if return_insect_line==False:
            return insect_point
        else:
            return insect_point,insect_line

    if len(obstacle.shape)==3:
        for obstacle_single in obstacle:
            for p in range(len(obstacle_single)):
                if p >= len(obstacle_single) - 1:
                    obstacleline = genline(obstacle_single[p], obstacle_single[0])
                    another_point = obstacle_single[0]
                else:
                    obstacleline = genline(obstacle_single[p], obstacle_single[p + 1])
                    another_point = obstacle_single[p + 1]

                if is_intersecting(line, obstacleline) and is_point_on_segment(another_point, line) == False:
                    insect_point.append(line_line_intersection(line, obstacleline))
                    insect_line.append(obstacleline)

        if return_insect_line == False:
            return insect_point
        else:
            return insect_point, insect_line


def point_in_obstacles_judge(point,obstacle,outside_point=[0,0],edge=False): #判断线段与多个多边形是否有交点
    line=genline(point,outside_point)

    if (type(obstacle).__name__ == 'list'):
        obstacle=np.array(obstacle)

    if len(obstacle.shape)==2:
        intersect_cout = 0
        for p in range(len(obstacle)):

            if p>=len(obstacle)-1:
                obstacleline = genline(obstacle[p], obstacle[0])
                another_point=obstacle[0]
            else:
                obstacleline = genline(obstacle[p], obstacle[p+1])
                another_point=obstacle[p+1]
            # print("当前角", obstacle[p],"another point",another_point)
            if is_intersecting(line, obstacleline) and is_point_on_segment(another_point,line)==False:
                # print("只与当前角和线相交")
                if  is_point_on_segment(obstacle[p],line)==False:
                    # print("当前角不在线上")
                    if edge==True:
                        # print("边界算入")
                        if is_point_on_segment(point,obstacleline)==False:
                            # print("点不在线上")
                            intersect_cout += 1
                        else:
                            return True
                    else:
                        # print("边界不算入")
                        if is_point_on_segment(point,obstacleline)==False:
                            # print("点不在线上")
                            intersect_cout += 1
                        else:
                            return False
                else:
                    # print("当前角在线上")
                    if point[0]==obstacleline[0] and point[1]==obstacleline[1]: #判断是否与顶点重合
                        return False
                    else:
                        line_point0 = obstacle[p-1]
                        line_point1 = [obstacleline[0], obstacleline[1]]
                        line_point2 = [obstacleline[2], obstacleline[3]]
                        angle_point0=cal_ang(line_point0,line_point1,point)
                        angle_point1=cal_ang(line_point2, line_point1, point)
                        angle_corner=cal_ang(line_point0,line_point1,line_point2)


                        angle_point3 = cal_ang(line_point0, line_point1, outside_point)
                        angle_point4 = cal_ang(line_point2, line_point1, outside_point)
                        # print(angle_corner,angle_point0,angle_point1,angle_point3,angle_point4)

                        if angle_point1<angle_corner and angle_point0<angle_corner:
                            if angle_point3>angle_corner or angle_point4>angle_corner:
                                # print("符合条件")
                                intersect_cout+=1
                        elif angle_point3<angle_corner and angle_point4<angle_corner:
                            if angle_point1>angle_corner or angle_point0>angle_corner:
                                # print("符合条件")
                                intersect_cout += 1

        if intersect_cout%2==1:
            return True
        else:
            return False

    if len(obstacle.shape)==3:
        for obstacle_single in obstacle:
            intersect_cout = 0
            for p in range(len(obstacle_single)):
                if p >= len(obstacle_single) - 1:
                    obstacleline = genline(obstacle_single[p], obstacle_single[0])
                    another_point = obstacle_single[0]
                else:
                    obstacleline = genline(obstacle_single[p], obstacle_single[p + 1])
                    another_point = obstacle_single[p + 1]
                # print("当前角", obstacle[p],"another point",another_point)
                if is_intersecting(line, obstacleline) and is_point_on_segment(another_point, line) == False:
                    # print("只与当前角和线相交")
                    if is_point_on_segment(obstacle_single[p], line) == False:
                        # print("当前角不在线上")
                        if edge == True:
                            # print("边界算入")
                            if is_point_on_segment(point, obstacleline) == False:
                                # print("点不在线上")
                                intersect_cout += 1
                            else:
                                return True
                        else:
                            # print("边界不算入")
                            if is_point_on_segment(point, obstacleline) == False:
                                # print("点不在线上")
                                intersect_cout += 1
                            else:
                                return False
                    else:
                        # print("当前角在线上")
                        if point[0] == obstacleline[0] and point[1] == obstacleline[1]:  # 判断是否与顶点重合
                            return False
                        else:
                            line_point0 = obstacle_single[p - 1]
                            line_point1 = [obstacleline[0], obstacleline[1]]
                            line_point2 = [obstacleline[2], obstacleline[3]]
                            angle_point0 = cal_ang(line_point0, line_point1, point)
                            angle_point1 = cal_ang(line_point2, line_point1, point)
                            angle_corner = cal_ang(line_point0, line_point1, line_point2)

                            angle_point3 = cal_ang(line_point0, line_point1, outside_point)
                            angle_point4 = cal_ang(line_point2, line_point1, outside_point)
                            # print(angle_corner,angle_point0,angle_point1,angle_point3,angle_point4)

                            if angle_point1 < angle_corner and angle_point0 < angle_corner:
                                if angle_point3 > angle_corner or angle_point4 > angle_corner:
                                    # print("符合条件")
                                    intersect_cout += 1
                            elif angle_point3 < angle_corner and angle_point4 < angle_corner:
                                if angle_point1 > angle_corner or angle_point0 > angle_corner:
                                    # print("符合条件")
                                    intersect_cout += 1
            if intersect_cout % 2 == 1:
                return True

        return False

def fractional_vectors(ray,line,func=None):#在方向上的值或分向量  ray向量[B,A] A为终点B为起点 line为方向或向量[B,A]
    line=[line[0]/math.sqrt(line[0]**2+line[1]**2),line[1]/math.sqrt(line[0]**2+line[1]**2)]
    num=ray[0]*line[0]+ray[1]*line[1]
    if func==None:
        return num
    else:
        line=[line[0]*num,line[1]*num]
        return line
