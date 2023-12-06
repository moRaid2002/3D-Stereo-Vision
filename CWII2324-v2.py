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


def triangulate_point2(p_left, p_right, M_left, M_right, K):
    # Convert image points to normalized camera coordinates
    K_inv = np.linalg.inv(K)
    p_left_norm = K_inv.dot(np.append(p_left, 1))
    p_right_norm = K_inv.dot(np.append(p_right, 1))

    # Construct matrix A for the linear system Ap = 0
    A = np.array([
        p_left_norm[0] * M_left[2, :] - M_left[0, :],
        p_left_norm[1] * M_left[2, :] - M_left[1, :],
        p_right_norm[0] * M_right[2, :] - M_right[0, :],
        p_right_norm[1] * M_right[2, :] - M_right[1, :]
    ])

    # Solve for P using SVD
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1]
    P = P / P[3]  # Convert to homogeneous coordinates

    return P[:3]


def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def compute_fundamental_matrix(M_left, M_right, K):
    # Compute Left-to-Right Camera Transformation
    M_left_to_right = np.dot(M_right, np.linalg.inv(M_left))

    # Extract R and T from the Left-to-Right Camera Transformation
    R = M_left_to_right[:3, :3]
    T = M_left_to_right[:3, 3]

    # Compute Essential Matrix
    T_skew = skew_symmetric(T)
    E = np.dot(T_skew, R)

    # Compute Fundamental Matrix
    K_inv = np.linalg.inv(K)
    F = np.dot(np.dot(K_inv.T, E), K_inv)

    return F, R, T


def remove_duplicates_with_higher_distance(points_list):
    best_entries = {}

    for c, mc, d in points_list:
        if c not in best_entries or d < best_entries[c][1]:
            best_entries[c] = (mc, d)

    cleaned_list = [(c, mc, d) for c, (mc, d) in best_entries.items()]
    return cleaned_list


def findCircles(img, imageNumber):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    blur = cv2.medianBlur(img_gray, 5)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=17, minRadius=10,
                               maxRadius=100)

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

    F, R, T = compute_fundamental_matrix(H0_wc, H1_wc, K.intrinsic_matrix)

    epipolarLines = []
    epipolarLinesR = []
    for point in circles_center0:
        homogeneous_point = np.array([point[0], point[1], 1])
        homogeneous_pointR = np.array([point[0], point[1] - point[2], 1])

        epipolar_line = np.dot(F, homogeneous_point)
        epipolar_lineR = np.dot(F, homogeneous_pointR)

        epipolar_line_homogeneous = np.array([epipolar_line[0], epipolar_line[1], epipolar_line[2]])
        epipolar_line_homogeneousR = np.array([epipolar_lineR[0], epipolar_lineR[1], epipolar_lineR[2]])

        epipolar_line_homogeneous /= np.linalg.norm(epipolar_line_homogeneous)
        epipolar_line_homogeneousR /= np.linalg.norm(epipolar_line_homogeneousR)

        a, b, c = epipolar_line_homogeneous
        x0, x1 = 0, img1.shape[1] - 1
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        epipolarLines.append((a, b, c, point))

        epipolar_line_image = cv2.line(img1, (x0, y0), (x1, y1), (255, 0, 0), 2)

        a, b, c = epipolar_line_homogeneousR
        x0, x1 = 0, img1.shape[1] - 1
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        epipolarLinesR.append((a, b, c, point))

        epipolar_line_image = cv2.line(img1, (x0, y0), (x1, y1), (0, 0, 255), 2)
        new = cv2.circle(img0, (point[0], point[1] - point[2]), 1, (0, 0, 0), 2)

    # Step 7: Display the updated second image with the drawn epipolar line
    cv2.imshow('Epipolar Line Image', epipolar_line_image)
    cv2.imwrite("lines.png", epipolar_line_image)
    cv2.imwrite("r.png", new)

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

        if mind < 0.25:
            match.append((matchCircle, circle, mind))

    match = remove_duplicates_with_higher_distance(match)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255), (0, 128, 255)]
    i = 0
    for c, mc, d in match:
        if i < 6:
            color = colors[i]
            i = i + 1
            match1 = cv2.circle(img1_copy, (mc[0], mc[1]), 2, color, 3)
            match0 = cv2.circle(img0_copy, (c[0], c[1]), 2, color, 3)

    cv2.imwrite("match0.png", match0)
    cv2.imwrite("match1.png", match1)

    matchR = []
    for a, b, c, point in epipolarLinesR:

        matchedpoint = point
        for lc, rc, d in match:
            if lc == point:
                matchedpoint = rc
        distances = []
        for theta in range(0, 360):
            x0 = matchedpoint[0] + matchedpoint[2] * np.cos(np.radians(theta))
            y0 = matchedpoint[1] + matchedpoint[2] * np.sin(np.radians(theta))
            possible_R = (x0, y0)
            dis = abs(a * x0 + b * y0 + c) / (a ** 2 + b ** 2) ** 0.5
            distances.append((possible_R, dis))
        mind = sys.maxsize
        bestpoint = matchedpoint

        for pR, dR in distances:

            if dR < mind:
                mind = dR
                bestpoint = pR

        matchR.append(((point[0], point[1] - point[2]), bestpoint))

    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    ###################################
    # Implement Task 6
    spheres_3D_world = []

    for lc, rc, _ in match:
        P = triangulate_point2(np.array([lc[0], lc[1]]), np.array([rc[0], rc[1]]), H0_wc, H1_wc, K.intrinsic_matrix)

        if -12 * 1.2 <= P[0] <= 12 * 1.2 and 1 * 0.8 <= P[1] <= 1.6 * 1.2 and -6 * 1.2 <= P[2] <= 6 * 1.2:
            spheres_3D_world.append(P)

    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################
    errors = []
    for c in spheres_3D_world:
        maxd = sys.maxsize
        p = c
        for gt in np.asarray(pcd_GTcents.points):
            d = np.linalg.norm(c - gt)
            if d < maxd:
                maxd = d
                p = gt
        errors.append((c,p,maxd))
    index = 1
    for est_c , c , e in errors:
        print("-------------------------SPHERE "+str(index)+"--------------------------------")
        print("the ground truth coordinates of the sphere center is: " + str(c) + " \nthe estimated center : " + str(est_c)+ " \nwith an error of: " + str(e))
        index +=1

    print("-----------------------------------------------------------------")
    average = sum(e for a, b, e in errors)/ len(errors)
    print("Average error  across all spheres is:  " + str(average))
    print("-----------------------------------------------------------------")
    pcd_est_cents = o3d.geometry.PointCloud()
    pcd_est_cents.points = o3d.utility.Vector3dVector(np.array(spheres_3D_world)[:, :3])
    pcd_est_cents.paint_uniform_color([0., 0., 1.])


    # Add the point clouds to the visualization
    vis = o3d.visualization.Visualizer()

    vis.create_window(width=640, height=480, left=0, top=0)
    for m in [obj_meshes[0], pcd_GTcents]:
        vis.add_geometry(m)

    vis.add_geometry(pcd_est_cents)

    vis.run()
    vis.destroy_window()
    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################
    radius_3D_world = []

    for lr, rr in matchR:
        P = triangulate_point2(np.array([lr[0], lr[1]]), np.array([rr[0], rr[1]]), H0_wc, H1_wc, K.intrinsic_matrix)

        if -12 * 2 <= P[0] <= 12 * 2 and 1 * 0.25 <= P[1] <= 1.6 * 2 and -6 * 2 <= P[2] <= 6 * 2:
            radius_3D_world.append(P)


    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################
    vis = o3d.visualization.Visualizer()

    vis.create_window(width=640, height=480, left=0, top=0)
    vis.add_geometry(obj_meshes[0])
    for cen in np.asarray(pcd_GTcents.points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cen[1])
        sphere.translate(cen)
        sphere.paint_uniform_color([0., 0., 0.])
        vis.add_geometry(sphere)

    for c3D in spheres_3D_world:
        for r3D in radius_3D_world:
            # Create a sphere geometry at the origin
            radius = np.linalg.norm(c3D - r3D)
            if 1 * 0.5 <= radius <= 1.6 * 1.5:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(c3D)
                sphere_cloud = sphere.sample_points_poisson_disk(number_of_points=1000)

                colors = np.asarray(sphere_cloud.colors)
                red_color = [1, 0, 0]  # Red color
                colors = np.tile(red_color, (colors.shape[0], 1))

                # Update the colors of the point cloud
                sphere_cloud.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(sphere_cloud)

    vis.run()
    vis.destroy_window()

    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################
