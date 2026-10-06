import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
import math
import numpy as np
import scipy.optimize as opt
import math
rng = np.random.default_rng()



class Environment3D:
    def __init__(self):
        self.boundary = []
        self.blocks = []
        self.start_point = None
        self.goal_point = None
        self.safety_margin = 0.25  # Safety margin around obstacles

    def set_start_goal_points(self, start=None, goal=None):
        scaleX = (self.boundary[3] - self.boundary[0])
        scaleY = (self.boundary[4] - self.boundary[1])
        scaleZ = (self.boundary[5] - self.boundary[2])
        try:
            if(start is None):
                self.start_point = self.generate_random_free_point()
            else:
                self.start_point = start
            if(goal is None):
                self.goal_point = self.generate_random_free_point()
            else:
                self.goal_point = goal
            return True
        except:
            return False


      
    def parse_map_file(self, filename):
        """
        Parse the map file and extract boundary and blocks
        coords = [xmin, ymin, zmin, xmax, ymax, zmax]
        colors = [r, g, b] each in [0, 1] (make sure color values are in range 0-1)
        self.blocks.append((coords, colors))
        self.boundary = [xmin, ymin, zmin, xmax, ymax, zmax]
        return True if successful, False otherwise (True if file was parsed successfully, without any error.)
        """
        
        with open(filename, 'r') as f:
            #print("WOOH")
            lines = f.readlines()
            #print("fail?")
            for line in lines:
                res = re.split(r"[\s]+", line)
                if('boundary' in res[0]):
                    res = list(map(lambda x: float(x.replace('\n', '')), res[1:7]))
                    self.boundary = res
                elif('block' in res[0]):
                    coords = list(map(lambda x: float(x.replace('\n', '')), res[1:7]))
                    colors = list(map(lambda x: float(x.replace('\n', '')), res[7:10]))
                    self.blocks.append(tuple((coords, colors)))
        self.set_start_goal_points()
        return True
    
        return False

    def gen_eqs(self, block):

        planes = self.get_six_eqs(block[0])
        return planes.copy()


    def get_six_eqs(self, coords):
        xmin, ymin, zmin, xmax, ymax, zmax = coords
        x = [xmin, xmax]
        y = [ymin, ymax]
        z = [zmin, zmax]
        #print(f"X: {x}, Y: {y}, Z: {z}")

        points = []
        planes = []
        for i in range(0, 3):
            #First three points.
            if(i == 0):
                #Constrain x
                for j in range(0, 2):
                    points.append([x[0], y[j], z[j]])
                points.append([x[0], y[0], z[1]])
                planes.append(points.copy())
                points.clear()
                for j in range(0, 2):
                    points.append([x[1], y[j], z[j]])
                points.append([x[1], y[0], z[1]])
                planes.append(points.copy())
                points.clear()
            elif(i == 1):
                for j in range(0, 2):
                    points.append([x[j], y[0], z[j]])
                points.append([x[0], y[0], z[1]])
                planes.append(points.copy())
                points.clear()
                for j in range(0, 2):
                    points.append([x[j], y[1], z[j]])
                points.append([x[0], y[1], z[1]])
                planes.append(points.copy())
                points.clear()
            else:
                for j in range(0, 2):
                    points.append([x[j], y[j], z[0]])
                points.append([x[0], y[1], z[0]])
                planes.append(points.copy())
                points.clear()
                for j in range(0, 2):
                    points.append([x[j], y[j], z[1]])
                points.append([x[0], y[1], z[1]])
                planes.append(points.copy())
                points.clear()
        #print(f"Planes: {planes}")
        return planes
            



    ##############################################
    #### TODO - Implement collision checking #####
    ##############################################
    def is_point_in_free_space(self, point):
        """
        Check if a point is in free space (not inside any obstacle)
        Complete implementation with collision checking
        return True if free, False if in collision
        """
        #Accept a single point (3,) or a batch of points (N, 3).
        pts = np.atleast_2d(np.asarray(point, dtype=float))
        if(self.is_point_in_boundary(pts) == False):
            return False
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        m = self.safety_margin
        for block_and_color in self.blocks:
            xmin, ymin, zmin, xmax, ymax, zmax = block_and_color[0]
            in_block = ((px >= xmin - m) & (px <= xmax + m) &
                        (py >= ymin - m) & (py <= ymax + m) &
                        (pz >= zmin - m) & (pz <= zmax + m))
            if in_block.any():
                return False
        return True

    def is_point_in_boundary(self, point):
        """
        True if point is within boundary plus some safety_margin.
        False if point is beyond bondary. """
        pts = np.atleast_2d(np.asarray(point, dtype=float))
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        m = self.safety_margin
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        inside = ((px >= xmin + m) & (px <= xmax - m) &
                  (py >= ymin + m) & (py <= ymax - m) &
                  (pz >= zmin + m) & (pz <= zmax - m))
        return bool(inside.all())
            

    def is_collision_free(self, p1, p2, num_checks = 100):
        
        if(self.is_point_in_free_space(p1) == False or self.is_point_in_free_space(p2) == False):
            print("Returning False Here.")
            return False

        for block in self.blocks:
            xmin, ymin, zmin, xmax, ymax, zmax = block[0]
            bounds = [
                (xmin, xmax),
                (ymin, ymax),
                (zmin, zmax)
            ]
            planes = self.gen_eqs(block)
            print(f"Len: {len(planes)}")
            initial_guess = np.array(p1)+(.5*(np.array(p2)-np.array(p1)))
            print(f"Initial Guess: {initial_guess}")
            
            for plane in planes:
                res, point, t = self.line_plane(p1, p2, plane, mode = "segment", eps = 1e-12)
                if res:
                    if(self.is_point_in_free_space(point)):
                        print(f"Point was in free space so ignore: {point}")
                        continue
                    else:
                        return initial_guess, point, False
                '''
                constants = (plane, p1, p2)
                result = opt.minimize(objective, initial_guess, bounds = bounds, args = constants)
                if(result.success):
                    print(f"Result Success: {result.fun}")
                    if(result.fun < 1e-6):
                        return initial_guess, result.x, False
                '''
        return initial_guess, [0, 0, 0], True
    

    ##############################################
    #### TODO - Implement line - collision checking #####
    ##############################################
    def is_line_collision_free(self, p1, p2, num_checks=100):
        """
        Check if a line segment between two points is collision-free
        Used for RRT* edge validation
        return True if free, False if in collision
        """

        
        #print(f"First Point: {p1}, second point: {p2}")
        #First simply check two points.
        if(self.is_point_in_free_space(p1) == False or self.is_point_in_free_space(p2) == False):
            #print("Returning False Here.")
            return False
        
        else:
            p1x, p1y, p1z = p1
            p2x, p2y, p2z = p2
            dist_vec = np.abs(np.subtract(p2, p1))
            total_dist = np.linalg.norm(dist_vec)
            xDir = 1
            yDir = 1
            zDir = 1
            if(p1x >= p2x):
                xDir = -1
            if(p1y >= p2y):
                yDir = -1
            if(p1z >= p2z):
                zDir = -1
            #num checks per one meter, i.e. 20 checks is a point every 5 cm. 
            
            x_check, y_check, z_check = np.divide(dist_vec, num_checks)
            #print(f"Check Dist: {[x_check, y_check, z_check]}")
            #print(f"Core Dist: {[x_core, y_core, z_core]}")

            step = np.array([xDir*x_check, yDir*y_check, zDir*z_check])
            steps = np.arange(1, num_checks + 1).reshape(-1, 1)
            allPoints = np.array(p1) + steps * step
            return bool(self.is_point_in_free_space(allPoints))
        
                            

    def visualize_environment(self, ax=None, show_start_goal = False):
        """Visualize the environment"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False
            ax.set_xlim(self.boundary[0], self.boundary[0+3])
            ax.set_ylim(self.boundary[1], self.boundary[1+3])
            ax.set_zlim(self.boundary[2], self.boundary[2+3])
            ax.set_aspect('equal')

        verts = np.array([])
        colors = []
        for block_color in self.blocks:
            coords = block_color[0]
            color = np.divide(block_color[1], 255)
            if(len(verts) == 0):
                verts = self.list_of_coords(coords)
            else:
                verts = np.append(verts, self.list_of_coords(coords), axis = 0)
            for i in range(6): #six sides
                colors.append(color)
        if (show_start_goal):
            ax.scatter(self.start_point[0], self.start_point[1], self.start_point[2], s=100, color= 'red', marker = '*')
            ax.scatter(self.goal_point[0], self.goal_point[1], self.goal_point[2], s=100, color= 'blue', marker = 'X')

        poly = Poly3DCollection(verts, alpha = .9)
        poly.set_facecolor(colors)
        poly.set_edgecolor('black')
        ax.add_collection(poly)
        if standalone:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Environment')
            ax.set_xlim(self.boundary[0], self.boundary[0+3])
            ax.set_ylim(self.boundary[1], self.boundary[1+3])
            ax.set_zlim(self.boundary[2], self.boundary[2+3])
            ax.set_aspect('equal') 
            ax.legend()
            plt.tight_layout()
            plt.show()

        return ax
        
    def list_of_coords(self, coords):
        xmin, ymin, zmin, xmax, ymax, zmax = coords
        verts = []
        for ax in 'xyz':
            if(ax == 'x'):
                face1 = [(xmin, ymin, zmin), (xmin, ymin, zmax), (xmin, ymax, zmax), (xmin, ymax, zmin), (xmin, ymin, zmin)]
                face2 = [(xmax, ymin, zmin), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmax, ymax, zmin), (xmax, ymin, zmin)]
            elif(ax == 'y'):
                face1 = [(xmin, ymin, zmin), (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymin, zmin), (xmin, ymin, zmin)]
                face2 = [(xmin, ymax, zmin), (xmin, ymax, zmax), (xmax, ymax, zmax), (xmax, ymax, zmin), (xmin, ymax, zmin)]
            else:
                face1 = [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymin, zmin), (xmin, ymin, zmin)]
                face2 = [(xmin, ymin, zmax), (xmin, ymax, zmax), (xmax, ymax, zmax), (xmax, ymin, zmax), (xmin, ymin, zmax)]    
            verts.append(face1)
            verts.append(face2)  
        return np.asarray(verts)

    
    
    def generate_random_free_point(self):
        """
        Generate a random point in free space
        Used for RRT* sampling
        """
        if not self.boundary:
            return None
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(xmin + self.safety_margin, xmax - self.safety_margin)
            y = np.random.uniform(ymin + self.safety_margin, ymax - self.safety_margin)
            z = np.random.uniform(zmin + self.safety_margin, zmax - self.safety_margin)
            
            point = [x, y, z]
            if self.is_point_in_free_space(point):
                return point
        
        print("Warning: Could not generate random free point after", max_attempts, "attempts")
        return None

    def get_environment_info(self):
        """Get information about the environment layout"""
        if not self.boundary:
            return "No boundary defined"
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        
        info = f"""
        Environment Information:
        Boundary: [{xmin}, {ymin}, {zmin}] to [{xmax}, {ymax}, {zmax}]
        Size: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f} meters
        Volume: {(xmax-xmin)*(ymax-ymin)*(zmax-zmin):.1f} cubic meters
        Obstacles: {len(self.blocks)} blocks
        Safety margin: {self.safety_margin} meters
        """
        
        if self.start_point and self.goal_point:
            distance = np.linalg.norm(np.array(self.goal_point) - np.array(self.start_point))
            info += f"  Start-Goal distance: {distance:.2f} meters\n"
        
        return info

    def line_plane(self, A, B, plane, mode="line", eps=1e-12):
        """
        mode: "line" (infinite), "ray" (t >= 0), or "segment" (0 <= t <= 1)
        Returns (status, point, t) where status is one of:
        "hit", "miss", "parallel", "coplanar"
        """
        p1, p2, p3 = plane
        n, c = self.plane_from_points(p1, p2, p3)
        
        A = np.asarray(A, float); B = np.asarray(B, float)
        n = np.asarray(n, float)
        d = B - A

        nn = np.linalg.norm(n)
        n, c = n / nn, c / nn                    # makes distances true distances

        denom = np.dot(n, d)
        dist = c - np.dot(n, A)                  # signed distance from A to plane

        if abs(denom) < eps * np.linalg.norm(d):
            return False

        t = dist / denom
        if mode == "ray" and t < 0:
            return False, [None, None, None], None
        if mode == "segment" and not (0 <= t <= 1):
            return False, [None, None, None], None
        return True, A + t * d, t

    def plane_from_points(self, P1, P2, P3, normalize=True, eps=1e-12):
        P1 = np.asarray(P1, float)
        P2 = np.asarray(P2, float)
        P3 = np.asarray(P3, float)

        n = np.cross(P2 - P1, P3 - P1)
        norm = np.linalg.norm(n)

        # Degenerate if the points are collinear (or coincident)
        scale = max(np.linalg.norm(P2 - P1), np.linalg.norm(P3 - P1))
        if norm < eps * max(scale**2, 1.0):
            raise ValueError("Points are collinear; no unique plane")

        if normalize:
            n = n / norm
        c = np.dot(n, P1)
        return n, c