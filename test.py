from Code.environment import Environment3D
import cv2 
import matplotlib.pyplot as plt

print("HELLOOO")
env = Environment3D()
env.parse_map_file('maps/map1_2b.txt')

start_point = [-.25, -1, .35]
end_point = [5, 1, 1.5]
env.set_start_goal_points(start_point, end_point)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax = env.visualize_environment(ax)
flag = True

while True:
    # ... your frame reading and cv2.imshow() code ...
    p1 = env.generate_random_free_point()
    p2 = env.generate_random_free_point()
    po1 = ax.scatter(p1[0], p1[1], p1[2], s=100, color= 'green', marker = '*')
    po2 = ax.scatter(p2[0], p2[1], p2[2], s=100, color= 'blue', marker = 'X')
    l2,  = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    init_guess, point, collision = env.is_collision_free(p1, p2)
    coll = None
    if(collision):
        ax.set_title('Collision Free')
        p = ax.scatter(init_guess[0], init_guess[1], init_guess[2], color = 'black', marker = 'X')

    else:
        ax.set_title('Collision')
        print(f"Point of collision: {point}")
        coll = ax.scatter(point[0], point[1], point[2], color = 'red', marker = 'X', s = 100)
        p = ax.scatter(init_guess[0], init_guess[1], init_guess[2], color = 'black', marker = 'X')

    plt.show(block = False)

    if(input() == 'q'):
        break
    po1.remove()
    po2.remove()
    l2.remove()
    p.remove()
    if(coll is not None):
        coll.remove()
    ax.set_aspect('equal')


'''
p1 = [1, 0.0, .5]
p2 = [3, 0.0, .5]
po1 = ax.scatter(p1[0], p1[1], p1[2], s=100, color= 'green', marker = '*')
po2 = ax.scatter(p2[0], p2[1], p2[2], s=100, color= 'blue', marker = 'X')
l2,  = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
init_guess, point, collision = env.is_collision_free(p1, p2)
if(collision):
    ax.set_title('Collision Free')
    p = ax.scatter(init_guess[0], init_guess[1], init_guess[2], color = 'cyan', marker = 'X')
else:
    ax.set_title('Collision')
    print(f"Point of collision: {point}")
    ax.scatter(point[0], point[1], point[2], color = 'black', s = 1000, marker = 'X')
    p = ax.scatter(init_guess[0], init_guess[1], init_guess[2], s = 1000, color = 'cyan', marker = 'X')

plt.show()
'''