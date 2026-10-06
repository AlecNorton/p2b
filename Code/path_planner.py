import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random 
import math
class RRTNode:
    """Node for RRT* tree"""
    def __init__(self, position, parent=None):
        self.position = np.array(position, dtype=float)
        self.parent = parent
        self.cost = 0.0
        self.children = []

class PathPlanner:
    """
    Robust RRT* implementation for 3D path planning
    """
    
    def __init__(self, environment):
        self.env = environment
        self.waypoints = []
        self.tree_nodes = []
        
        # RRT* parameters
        self.max_iterations = 3000
        self.step_size = 1
        self.goal_radius = .5
        self.search_radius = 10
        self.goal_bias = 0.10  # 15% bias towards goal
        self.goal_node = None
        self.refinement_iterations = 1000
        self.refinement_flag = False

    def set_vars(self, max_iterations, step_size, goal_radius, search_radius, goal_bias):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.search_radius = search_radius
        self.goal_bias = goal_bias
    
    ############################################################################################################
    #### TODO - Implement RRT* path planning algorithm in 3D (use the provided environment class) ##############
    #### TODO - Store the final path in self.waypoints as a list of 3D points ##################################
    #### TODO - Add member functions as needed #################################################################
    ############################################################################################################

    def plan_path(self):
        self.env.visualize_environment(show_start_goal = True)
        print(f"Planning path between start: {self.env.start_point}, and goal: {self.env.goal_point}")
        self.tree_nodes = []
        self.tree_nodes.append(RRTNode(self.env.start_point))
        for k in range(self.max_iterations):
            #Generate a random point, ensuring they are NOT in an obstacle. 15% bias for goal
            goal_or_random = random()
            if(goal_or_random <= self.goal_bias):
                random_point = self.env.goal_point
            else:
                random_point = self.env.generate_random_free_point()
            #Determine closest node and generate a point in its step radius based on direction. 
            random_node = RRTNode(random_point)
            closest_node= self.find_closest_node(random_node)
            direction = np.subtract(random_node.position, closest_node.position)
            direction = direction/np.linalg.norm(direction)
            new_point = np.add(closest_node.position, np.multiply(direction, self.step_size))
            if(self.euclidian_dist(RRTNode(new_point), closest_node) > self.step_size):
                #print(f"dist: {self.euclidian_dist(RRTNode(new_point), closest_node)}")
                pass
            #If new point causes line collision, simply continue and just sample a new point. 

            if(self.env.is_line_collision_free(new_point, closest_node.position) == False):
                #print(f"Failed: {k}, new_point: {new_point}, closest_node: {closest_node.position}")
                continue
            new_node = RRTNode(new_point)
            self.parent_child(closest_node, new_node)
            #Find appropriate parent. 
            self.rewire(new_node)
            self.tree_nodes.append(new_node)
            #print(f"Distance: {np.linalg.norm(self.env.goal_point - new_node.position)}")
            if(np.linalg.norm(self.env.goal_point - new_node.position) <= self.goal_radius):

                #WE HAVE FOUND OUR PATH AND BREAK
                #print(f"New node parent: {new_node.parent}")
                ##waypoints = self.get_waypoints(new_node)
                self.goal_bias = 0.0
                #print(f"Waypoints: {waypoints}")
                #self.waypoints = list(waypoints)
                self.goal_node = new_node
                break
        if(self.goal_node is None):
            print(f"Found no node: {self.goal_node}, {k}")
            return False
        elif(self.refinement_flag):
            #Perform refinement stage. 
            for k in range(self.refinement_iterations):
                #Generate a random point, ensuring they are NOT in an obstacle. 15% bias for goal
                goal_or_random = random()
                if(goal_or_random <= self.goal_bias):
                    random_point = self.env.goal_point
                else:
                    random_point = self.env.generate_random_free_point()
                #Determine closest node and generate a point in its step radius based on direction. 
                random_node = RRTNode(random_point)
                closest_node= self.find_closest_node(random_node)
                new_point = self.steer(random_point, closest_node.position)
                if(self.euclidian_dist(RRTNode(new_point), closest_node) > self.step_size):
                    #print(f"dist: {self.euclidian_dist(RRTNode(new_point), closest_node)}")
                    pass
                #If new point causes line collision, simply continue and just sample a new point. 

                if(self.env.is_line_collision_free(new_point, closest_node.position) == False):
                    #print(f"Failed: {k}, new_point: {new_point}, closest_node: {closest_node.position}")
                    continue
                new_node = RRTNode(new_point)
                self.parent_child(closest_node, new_node)
                #Find appropriate parent. 
                self.rewire(new_node)
                self.tree_nodes.append(new_node)

                #Just in case we get a closer point than our goal node. 
                dist = np.linalg.norm(self.env.goal_point - new_node.position)
                if(dist <= self.goal_radius and dist <= np.linalg.norm(self.env.goal_point - self.goal_node.position)):
                    self.goal_node = new_node
            
            self.waypoints = self.extract_path(self.goal_node)
            print(f"Cost of refined path: {self.goal_node.cost}")
            
            return True
        else:
            self.waypoints = self.extract_path(self.goal_node)
            print(f"Cost of unrefined path: {self.goal_node.cost}")
            return True


    def steer(self, neighbor_point, sample_point):
        direction = np.subtract(sample_point, neighbor_point)
        direction = direction/np.linalg.norm(direction)
        new_point = np.add(neighbor_point, np.multiply(direction, self.step_size))
        return new_point
    
    def find_closest_node(self, new_node, tree=None):
        smallest_dist = -1
        closestNode = None
        if(tree is None):

            for node in self.tree_nodes:

                dist = self.euclidian_dist(new_node, node)
                if(dist < smallest_dist or smallest_dist == -1):
                    smallest_dist = dist
                    closestNode = node
        else:
            for node in tree:
                dist = self.euclidian_dist(new_node, node)
                if(dist < smallest_dist or smallest_dist == -1):
                    smallest_dist = dist
                    closestNode = node
        return closestNode


    def find_near_nodes(self, node1):
        neighboring_nodes = []
        for node2 in self.tree_nodes:
            if(self.euclidian_dist(node1, node2) <= self.search_radius):
                neighboring_nodes.append(node2)
        return neighboring_nodes

    def rewire(self, node1):
        neighboring_nodes = self.find_near_nodes(node1)
        #First "reparent" new node based on distance to starting point. 
        for neighbor in neighboring_nodes:
            if(neighbor.cost + self.euclidian_dist(node1, neighbor) < node1.cost):
                if(self.env.is_line_collision_free(neighbor.position, node1.position) == False):
                    #Collision so just ignore this one. 
                    continue
                #print("Reparenting.")
                self.parent_child(neighbor, node1)
        #Once done, node1 can become a parent of other neighboring nodes
        for neighbor in neighboring_nodes:
            if(node1.cost + self.euclidian_dist(node1, neighbor) < neighbor.cost):
                if(self.env.is_line_collision_free(neighbor.position, node1.position) == False):
                    continue
                self.parent_child(node1, neighbor)
    
    def is_path_valid(self, pos1, pos2):
        return self.env.is_line_collision_free(pos1, pos2)

    def choose_parent(self, neighboring_nodes, new_pos):
        best_parent = None
        best_cost = math.inf
        for neighbor in neighboring_nodes:
            new_cost = neighbor.cost + self.euclidian_dist(neighbor, RRTNode(new_pos))
            if(new_cost < best_cost):
                best_parent = neighbor
                best_cost = neighbor.cost + self.euclidian_dist(new_cost)
        return best_parent, best_cost
    


    def get_waypoints(self, end_node):

        
        #print(f"End Node: {end_node}")
        appending = [end_node]
        if(end_node.parent is None):
            #print("Node has no parent.")
            return np.array([end_node.position])
        else:
            return np.vstack([np.array([end_node.position]), self.get_waypoints(end_node.parent)])

    def extract_path(self, goal_node):
        waypoints = list(self.get_waypoints(goal_node))
        #print(f"Waypoints: {waypoints}")
        new_waypoints = []
        for i in reversed(range(len(waypoints))):
            pt = list(waypoints[i])
            new_waypoints.append(pt)
        return new_waypoints

    
    def euclidian_dist(self, node1: RRTNode, node2:RRTNode):
        pos1 = node1.position
        pos2 = node2.position
        #print(f"Pos1:{pos1}")
        #print(f"Pos2:{pos2}")
        return np.linalg.norm(pos1 -pos2)

    def parent_child(self, parentNode, childNode):
        if(childNode.parent is None):
            childNode.parent = parentNode
            childNode.cost = parentNode.cost + self.euclidian_dist(parentNode, childNode)
            parentNode.children.append(childNode)
        else:
            childNode.parent.children.remove(childNode)
            childNode.parent = parentNode
            childNode.cost = parentNode.cost + self.euclidian_dist(parentNode, childNode)
            parentNode.children.append(childNode)

    def simplify_path(self, waypoints):
        new_waypoints = waypoints.copy()
        
        for i in range(0, len(waypoints)-2):
            firstNode= waypoints[i]
            secondNode = waypoints[i+1]
            thirdNode = waypoints[i+2]
            dist = np.subtract(np.array(firstNode), np.array(secondNode))
            dist = np.linalg.norm(dist)
            if(self.env.is_line_collision_free(firstNode, thirdNode, 1000*dist)):
                #Nothing stopping us from connecting directly. 
                new_waypoints.remove(secondNode)
                i += 1
        return new_waypoints


    
    def visualize_tree(self, ax=None):
        """Visualize the RRT* tree"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False
        
        # Draw tree edges
        for node in self.tree_nodes:
            if node.parent is not None:
                ax.plot([node.parent.position[0], node.position[0]],
                       [node.parent.position[1], node.position[1]],
                       [node.parent.position[2], node.position[2]],
                       'b-', alpha=0.3, linewidth=0.5)
        
        # Draw tree nodes
        if self.tree_nodes:
            positions = np.array([node.position for node in self.tree_nodes])
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='blue', s=10, alpha=0.6)
        
        # Draw final path
        if len(self.waypoints) > 0:
            waypoints = np.array(self.waypoints)
            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                   'ro-', markersize=8, linewidth=3, label='RRT* Path')
        
        if standalone:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('RRT* Tree and Path')
            ax.legend()
            plt.tight_layout()
            plt.show()
        
        return ax