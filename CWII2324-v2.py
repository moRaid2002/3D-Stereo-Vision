'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''
import sys

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse

'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''


def compute_3d_location(K, R, T, points2D):
    """
  K is the camera matrix that includes the focal length and the optical center.
  R and T are the rotation and translation of the camera.
  points2D is the 2D point in the camera view.
  """
    # Convert the 2D point to homogeneous coordinates
    points2D = np.array(points2D).reshape(-1, 1)
    points2D_homogeneous = np.hstack((points2D, np.ones((points2D.shape[0], 1))))

    # Compute the 3D location of the point in the camera view
    points3D_camera = np.linalg.inv(K) @ points2D_homogeneous

    # Compute the 3D location of the point in the world view
    points3D_world = np.linalg.inv(R @ T) @ points3D_camera

    return points3D_world


def find_AB(pl, pr, R, T):
    x = np.dot(np.transpose(R), pr)
    b = (pl[1] * T[2] - T[1]) / (x[1] - x[2] * pl[1])
    a = b * x[2] + T[2]

    return a, b


def findCircles(img, imageNumber):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    blur = cv2.medianBlur(img_gray, 5)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=15, minRadius=0,
                            maxRadius=180)

    centers = []
    if circles is not None:

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1], i[2])
            centers.append(center)
            # Draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imwrite("cirles" + str(imageNumber) + ".png", img)
    return centers


def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n, m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n, 1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3, :]
    new_points = new_points[:3, :].transpose()
    return new_points


def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__':

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6,
                        help='number of spheres')
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10,
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16,
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4,
                        help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8,
                        help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    args = parser.parse_args()

    if args.num <= 0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min >= args.sph_rad_max or args.sph_rad_min <= 0:
        print('invalid max and min sphere radii')
        exit()

    if args.sph_sep_min >= args.sph_sep_max or args.sph_sep_min <= 0:
        print('invalid max and min sphere separation')
        exit()

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=h, height=0.05, depth=w)
    box_H = np.array(
        [[1, 0, 0, -h / 2],
         [0, 1, 0, -0.05],
         [0, 0, 1, -w / 2],
         [0, 0, 0, 1]]
    )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2) / 10
        sph_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(args.sph_sep_min, args.sph_sep_max, 1)
        x = random.randrange(-h / 2 + 2, h / 2 - 2, step)
        z = random.randrange(-w / 2 + 2, w / 2 - 2, step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h / 2 + 2, h / 2 - 2, step)
            z = random.randrange(-w / 2 + 2, w / 2 - 2, step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1]))
        GT_rads.append(size)
        sph_H = np.array(
            [[1, 0, 0, x],
             [0, 1, 0, size],
             [0, 0, 1, z],
             [0, 0, 0, 1]]
        )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes + [coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')

    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45 * 5 + random.uniform(-5, 5)) / 180.
    H0_wc = np.array(
        [[1, 0, 0, 0],
         [0, np.cos(theta), -np.sin(theta), 0],
         [0, np.sin(theta), np.cos(theta), 20],
         [0, 0, 0, 1]]
    )

    # camera_1 (world to camera)
    theta = np.pi * (80 + random.uniform(-10, 10)) / 180.
    H1_0 = np.array(
        [[np.cos(theta), 0, np.sin(theta), 0],
         [0, 1, 0, 0],
         [-np.sin(theta), 0, np.cos(theta), 0],
         [0, 0, 0, 1]]
    )
    theta = np.pi * (45 * 5 + random.uniform(-5, 5)) / 180.
    H1_1 = np.array(
        [[1, 0, 0, 0],
         [0, np.cos(theta), -np.sin(theta), -4],
         [0, np.sin(theta), np.cos(theta), 20],
         [0, 0, 0, 1]]
    )
    H1_wc = np.matmul(H1_1, H1_0)

    render_list = [(H0_wc, 'view0.png', 'depth0.png'),
                   (H1_wc, 'view1.png', 'depth1.png')]

    #####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.

    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width = 640
    img_height = 480
    f = 415  # focal length
    # image centre in pixel coordinates
    ox = img_width / 2 - 0.5
    oy = img_height / 2 - 0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, f, f, ox, oy)

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
    ##################################################

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    ###################################
    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################
    img0_copy = cv2.imread('view0.png', -1)
    img1_copy = cv2.imread('view1.png', -1)
    circles_center0 = findCircles(img0, 0)
    circles_center1 = findCircles(img1, 1)

    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################
    # Step 2: Extract features/keypoints from both images
    # Use any feature detection algorithm of your choice
    # For example, using ORB feature detector
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img0, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img1, None)

    # Step 3: Compute the fundamental matrix
    # Use the keypoints and descriptors from step 2
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    fundamental_matrix, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    epipolarLines = []

    for point in circles_center0:
        homogeneous_point = np.array([point[0], point[1], 1])
        epipolar_line = np.dot(fundamental_matrix, homogeneous_point)

        epipolar_line_homogeneous = np.array([epipolar_line[0], epipolar_line[1], epipolar_line[2]])
        epipolar_line_homogeneous /= np.linalg.norm(epipolar_line_homogeneous)

        a, b, c = epipolar_line_homogeneous
        x0, x1 = 0, img1.shape[1] - 1
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        epipolarLines.append((a, b, c, point))

        epipolar_line_image = cv2.line(img1, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Step 7: Display the updated second image with the drawn epipolar line
    cv2.imshow('Epipolar Line Image', epipolar_line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    ###################################
    match = []
    lines = len(epipolarLines)
    for circle in circles_center1:
        distances = []
        for n in range(lines):
            first_circle = epipolarLines[n][3]

            distance = abs(circle[0] * epipolarLines[n][0] + circle[1] * epipolarLines[n][1] + epipolarLines[n][2])

            distances.append((distance, first_circle))
        mind = sys.maxsize
        matchCircle = circles_center0[0]
        for d, c in distances:
            if d < mind:
                mind = d
                matchCircle = c
        match.append((circle, matchCircle))

    for c, mc in match:
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        match0 = cv2.circle(img0_copy, (mc[0], mc[1]), 2, color, 3)
        match1 = cv2.circle(img1_copy, (c[0], c[1]), 2, color, 3)

    cv2.imwrite("match0.png", match0)
    cv2.imwrite("match1.png", match1)

    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    ###################################



    coordinates = []
    for x, y in match:
        coordinates.append(((x[0] - ox, x[1] - oy, 1), (y[0] - ox, y[1] - oy, 1)))

    points3D = []

    for left, right in coordinates:
        left3D = np.linalg.inv(K.intrinsic_matrix) @ left
        right3D = np.linalg.inv(K.intrinsic_matrix) @ right
        R = H1_wc[:3, :3]
        T = H1_wc[:3, 3]


        a, b = find_AB(left3D, right3D, R, T)




        proj1 = a * np.array(left)

        proj2 = b * np.dot(np.transpose(R), right3D) + T




        P = (proj1 + proj2) / 2



        points3D.append(P)
    print(points3D)

    E = np.matmul(np.transpose(K.intrinsic_matrix), np.matmul(fundamental_matrix, K.intrinsic_matrix))

    # Step 3: Decompose the essential matrix to obtain the relative camera pose
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K.intrinsic_matrix)
    projection_matrix = np.array([
        [2 * f / img_width, 0, 0, 0],
        [0, 2 * f / img_height, 0, 0],
        [0, 0, -1, 0]

    ])
    # Step 4: Compute the 3-D location of the sphere centers
    sphere_centers_3d = []
    for (circle_center0, circle_center1) in match:
        # Convert the circle centers to homogeneous coordinates
        circle_center0_homogeneous = np.array([circle_center0[0], circle_center0[1]])
        circle_center1_homogeneous = np.array([circle_center1[0], circle_center1[1]])

        # Triangulate the 3-D point using the relative camera pose
        P0 = H0_wc[:3]
        P1 = H1_wc[:3]
        print(P1)
        sphere_center_3d_homogeneous = cv2.triangulatePoints( P0,  P1, circle_center0_homogeneous,circle_center1_homogeneous)

        # Convert the 3-D point from homogeneous coordinates to Cartesian coordinates
        sphere_center_3d_cartesian = sphere_center_3d_homogeneous[:3] / sphere_center_3d_homogeneous[3]

        # Append the 3-D point to the list of sphere centers
        sphere_centers_3d.append((sphere_center_3d_cartesian[0][0], 1,sphere_center_3d_cartesian[1][0]))
    print(sphere_centers_3d)

    k = [[K.intrinsic_matrix[0][0],K.intrinsic_matrix[0][1],K.intrinsic_matrix[0][2],0],
         [K.intrinsic_matrix[1][0], K.intrinsic_matrix[1][1], K.intrinsic_matrix[1][2], 0],
         [K.intrinsic_matrix[2][0], K.intrinsic_matrix[2][1], K.intrinsic_matrix[2][2], 0],
         ]

    p = np.array(k)@H1_wc

    answer = np.array([[k[0][3]-k[2][3]],
              [k[1][3]-k[2][3]],
              [p[0][3]-p[2][3]],
              [p[1][3]-p[2][3]]])
    sphere_centers_3d = []
    for (circle_center0, circle_center1) in match:
        ur= circle_center1[0]
        vr = circle_center1[1]
        ul = circle_center0[0]
        vl = circle_center0[1]
        matrix = np.array([[ur*k[2][0] - k[0][0],     ur*k[2][1] - k[0][1],        ur*k[2][2] - k[0][2]   ],
                  [vr*k[2][0] - k[1][0],      vr*k[2][1] - k[1][1],     vr*k[2][2] - k[1][2]],
                  [ul * p[2][0]-p[0][0],      ul * p[2][1]-p[0][1],        ul * p[2][2]-p[0][2]],
                  [vl * p[2][0]-p[1][0],      vl * p[2][1]-p[1][1],        vl * p[2][2]-p[1][2]]])
        point_3D = np.linalg.inv( np.transpose(matrix) @ matrix ) @ np.transpose(matrix) @ answer

        sphere_centers_3d.append((point_3D[0][0], 1, point_3D[2][0]))


    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################

    pcd_est_cents = o3d.geometry.PointCloud()
    pcd_est_cents.points = o3d.utility.Vector3dVector(np.array(sphere_centers_3d)[:, :3])
    pcd_est_cents.paint_uniform_color([0., 0., 1.])
    print("--------")
    print(np.asarray(pcd_GTcents.points))
    print("--------")
    print(np.asarray(pcd_est_cents.points))
    print("--------")
    print(np.array(GT_rads))
    # Add the point clouds to the visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, left=0, top=0)
    for m in [obj_meshes[0], pcd_GTcents]:

        vis.add_geometry(m)

    vis.add_geometry( pcd_est_cents)


    vis.run()
    vis.destroy_window()
    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################

    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################

    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################
