import predict
import determine_position
import numpy as np

def convert_frame(a, b, c, T):
    x = np.array([[a, b, c, 1]]).T
    y = (T @ x).squeeze()
    return y[0], y[1], y[2]

# load the trained model
# TODO determine path
predictor = predict.Predictor(ckpt_path='20000.pth')

# let the robot touch one edge
# 1st point on edge
N = 3 # take 3 points along an edge and take average
a1, b1, c1 = [0] * N, [0] * N, [0] * N
a2, b2, c2 = [0] * N, [0] * N, [0] * N

for i in range(N):
    depth_map0 = from_robot() # TODO (400,) ndarray, some assumed API
    a1[i], b1[i], c1[i], a2[i], b2[i], c2[i] = predictor.predict(depth_map0)
    a1[i], b1[i], c1[i], a2[i], b2[i], c2[i] = a1[i].item(), b1[i].item(), c1[i].item(), a2[i].item(), b2[i].item(), c2[i].item()

a1 = sum(a1) / N
b1 = sum(b1) / N
c1 = sum(c1) / N
a2 = sum(a2) / N
b2 = sum(b2) / N
c2 = sum(c2) / N

# convert them to world frame!
a1, b1, c1 = convert_frame(a1, b1, c1, camera_to_world) # camera_to_world: a transformation matrix
a2, b2, c2 = convert_frame(a2, b2, c2, camera_to_world)


# change to another edge
a3, b3, c3 = [0] * N, [0] * N, [0] * N
for i in range(N):
    depth_map0 = from_robot() # TODO (400,) ndarray, some assumed API
    _, _, _, a3[i], b3[i], c3[i] = predictor.predict(depth_map0)
    a3[i], b3[i], c3[i] = a3[i].item(), b3[i].item(), c3[i].item()

a3 = sum(a3) / N
b3 = sum(b3) / N
c3 = sum(c3) / N

a3, b3, c3 = convert_frame(a3, b3, c3, camera_to_world) # camera_to_world: a transformation matrix

T_WC = determine_position.calculate_T_WC(a1, b1, c1, a2, b2, c2, a3, b3, c3) # from cube to world

# hole position on cube = p, hole position in world = T_WC @ p