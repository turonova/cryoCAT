import numpy as np
import pandas as pd 
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
from cryocat import tgeometry as tg
from cryocat import cuboid_sampling
from cryocat import geom

# shape_layer is shape data layer, e.g. viewer.layers[x].data where z is layer id
def get_oversampling(shape_layer,sampling_distance):

    if isinstance(shape_layer,list):
    # create array from the list
        mask_points=np.concatenate( shape_layer, axis=0 )
    else:
        mask_points=shape_layer

    # get convex hull
    hull=ConvexHull(mask_points)

    tri_points=[]
    normals=[]

    for i in range(len(hull.simplices)):
        tp=hull.points[hull.simplices[i,:]]
        mp=tg.get_mesh(tp,sampling_distance)
        n_tp=hull.equations[i][0:3]

        for p in mp:
            if tg.point_inside_triangle(p,tp):
                tri_points.append(p)
                normals.append(n_tp)

    tri_points=np.array(tri_points)
    normals=np.array(normals)

    return tri_points, normals


# save shape layer into text file
def save_shapes(shape_layer, output_file_name):

    x=[]
    y=[]
    z=[]
    s_id=[]

    for i in range(len(shape_layer)):
        x.append(shape_layer[i][:,2])
        y.append(shape_layer[i][:,1])
        z.append(shape_layer[i][:,0])
        s_id.append(np.full(shape_layer[i].shape[0], i))

    x=np.concatenate( x, axis=0 )
    y=np.concatenate( y, axis=0 )
    z=np.concatenate( z, axis=0 )
    s_id=np.concatenate( s_id, axis=0 )

    # dictionary of lists  
    dict = {'s_id': s_id, 'x': x, 'y': y, 'z': z}

    df = pd.DataFrame(dict)
    # saving the dataframe 
    df.to_csv(output_file_name,index=False)


def load_shapes(input_file_name):
    df=pd.read_csv(input_file_name)
    s_id=df['s_id'].unique()

    shape_list=[]

    for sf in s_id:
        a=df.loc[df['s_id'] == sf]
        sp_array=np.zeros((a.shape[0],3))
        sp_array[:,2]=(a['x']).values
        sp_array[:,1]=(a['y']).values
        sp_array[:,0]=(a['z']).values
        shape_list.append(sp_array)

    return shape_list

def save_shapes_as_point_cloud(shape_layer,output_file_name):
    if isinstance(shape_layer,list):
        point_cloud=np.concatenate( shape_layer, axis=0 )
    else:
        point_cloud=shape_layer

    np.savetxt(output_file_name,point_cloud,fmt='%5.6f')


def load_shapes_as_point_cloud(input_file_name):
    point_cloud = np.loadtxt(input_file_name)
    return point_cloud


# visualize points    
def visualize_points(viewer,points, p_size):
    viewer.add_points(points, size=p_size)


# visualize normals
def visualize_normals(viewer,points, normals):
    vectors = np.zeros((points.shape[0], 2, 3), dtype=np.float32)
    vectors[:,0] = points
    vectors[:,1] = normals
    viewer.add_vectors(vectors, edge_width=1, length=10)

def expand_points(points, normals, distances, tb_distances = 0):
    """move sample points within a distances

    Args:
        points (ndarray): array of sample coordinates
        normals (ndarray): array of sample normals
        distances (int): moving distance in pixels
        tb_distances (int, optional): moving distance of top and bottom surfaces if needed. Defaults to 0 for no movement.

    Returns:
        cleaned_points (ndarray): array of expand and cleaned sample coordinates
        cleaned_normals (ndarray): array of expand and cleaned sample normals
    """
    moved_points = []
    tb_points = []
    for i in range(len(normals)):
        if abs(normals[i, 0]) == 1 and normals[i, 1] == 0 and normals[i, 2] == 0:
            tb_points.append(i)
        else:
            moved_points.append(i)
    points[moved_points] = points[moved_points] + distances*normals[moved_points]

    if tb_distances == 0: # 0 for nonmove top and bottom
        cleaned_points = points + distances*normals
        cleaned_normals = normals
    elif tb_distances != 0: # other value for move top and bottom with extra distance
        points[tb_points] = points[tb_points] + (distances+tb_distances)*normals[tb_points]

    # removing the distinct points after shifting
    drop_lim = [max(points[tb_points][:, 0]), min(points[tb_points][:, 0])]
    drop_list = [i for i in range(len(points)) if points[i, 0] > drop_lim[0] or points[i, 0] < drop_lim[1]]
    cleaned_points = np.delete(points, drop_list, axis=0)
    cleaned_normals = np.delete(normals, drop_list, axis=0)
    
    return cleaned_points, cleaned_normals

# remove sample points with specific normals value
def rm_points(points, normals, rm_surface):
    # this function only consider removing the top and bottom surface
    # TODO more precise removing method
    # 1 for top, -1 for bottom, 0 for both
    # face with smaller z value is bottom face, bigger z value is the top face
    removed_points = []
    if rm_surface == 1:
        tob = 1
    elif rm_surface == 0:
        tob = 1
        normals[:,0] = np.absolute(normals[:,0])
    elif rm_surface == -1:
        tob = -1

    for i in range(len(normals)):
        if normals[i, 0] == tob and normals[i, 1] == 0 and normals[i, 2] == 0:
            removed_points.append(i)

    cleaned_points = np.delete(points, removed_points, axis=0)
    cleaned_normals = np.delete(normals, removed_points, axis=0)

    return cleaned_points, cleaned_normals

# replace the normal with the closest postion from convexHull
def reset_normals (samp_dist, shift_dist, tb_move, shapes_data, motl, bin_factor):
    # get the oversampling
    tri_points, tri_normals=cuboid_sampling.get_oversampling(shapes_data, samp_dist)
    # shifting points to the normal direction
    tri_points, tri_normals = cuboid_sampling.expand_points(tri_points, tri_normals, shift_dist, tb_move)  # negative for shifting in opposite direction

    #reorganize x,y,z into z,y,x to match with tri_points
    motl_points = np.flip(motl.values[:,7:10], axis=1)
    sample_points = tri_points*bin_factor # add binning to tri_points
    #searching for the closest point to motl_points
    kdtree = KDTree(sample_points)
    dist,points = kdtree.query(motl_points,1)

    ## replacing normals in motlist to new normals
    n_normals = tri_normals[points]
    # create panda frames from normals
    pd_normals = pd.DataFrame({'x':n_normals[:,2],'y':n_normals[:,1],'z':n_normals[:,0]}) 
    # get Euler angles from coordinates
    phi,psi,theta=tg.normals_to_euler_angles(pd_normals)
    # create pandas from angles
    pd_angles=pd.DataFrame({'phi':phi,'psi':psi,'theta':theta})
    # replace angles
    motl[['phi','psi','theta']]=pd_angles.values

    return motl

def get_sampling_pandas(shapes_data, overSample_dist, shift_dist = None, tb_dist = 0, rm_surface = None):
    # get the oversampling
    tri_points, tri_normals=cuboid_sampling.get_oversampling(shapes_data, overSample_dist)
    # shifted points to the normal direction
    if shift_dist !=  None:
        tri_points, tri_normals = cuboid_sampling.expand_points(tri_points, tri_normals, shift_dist, tb_dist)  # negative for shifting in opposite direction, set 4 argument to 0 to omit moving of top and bottom layer.
    # remove unecessary points base on condition
    if rm_surface != None:
        tri_points, tri_normals = cuboid_sampling.rm_points(tri_points, tri_normals, rm_surface)
    # create panda frames from points
    pd_points = pd.DataFrame({'x':tri_points[:,2],'y':tri_points[:,1],'z':tri_points[:,0]}) 
    # create panda frames from normals
    pd_normals = pd.DataFrame({'x':tri_normals[:,2],'y':tri_normals[:,1],'z':tri_normals[:,0]})
    # get Euler angles from coordinates
    phi,psi,theta=tg.normals_to_euler_angles(pd_normals)
    # create pandas 
    pd_angles=pd.DataFrame({'phi':phi,'psi':psi,'theta':theta})

    return pd_points, pd_angles

def get_surface_area_from_hull(mask_points, rm_faces):
    #get convexhull
    hull=ConvexHull(mask_points)
    faces = hull.equations

    # find indices of surface that were belongs to top or bottom
    if rm_faces == 0:
        updated_area = hull.area
    else: 
        if rm_faces == 1:
            tb_faces = [i for i, num in enumerate(faces) if sum(num[0:3] == [1,0,0]) == 3]
        elif rm_faces == -1:
            tb_faces = [i for i, num in enumerate(faces) if sum(num[0:3] == [-1,0,0]) == 3]
        elif rm_faces == 0:
            tb_faces = [i for i, num in enumerate(faces) if sum(abs(num[0:3]) == [1,0,0]) == 3]
        if tb_faces == []:
            raise ValueError("The target top/bottom surfaces doesn't exist")
        all_faces_points_in = hull.simplices # indices of points for surface
        tb_faces_points_in = all_faces_points_in[tb_faces]
        face_points_coord = [[mask_points[j] for j in i] for i in tb_faces_points_in]
        face_points_array = np.asarray(face_points_coord)
        tb_areas = geom.area_tri(face_points_array)
        total_tb_area = sum(tb_areas)
        updated_area = hull.area-total_tb_area 

    return updated_area