import numpy as np
import pandas as pd

from skimage import measure
from skimage import morphology

from cryocat import cryomaps
from cryocat import geom
from cryocat import visplot
from cryocat import cryomotl
from cryocat import cryomask


def compute_scores_map_threshold_triangle(scores_map):
    
    sp = np.sort(scores_map, axis=None)
    nbins = len(sp)

    # Find peak, lowest and highest gray levels.
    arg_peak_height = np.argmax(sp)
    peak_height = sp[arg_peak_height]
    arg_low_level, arg_high_level = np.where(sp > 0)[0][[0, -1]]

    # Flip is True if left tail is shorter.
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        sp = sp[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    # If flip == True, arg_high_level becomes incorrect
    # but we don't need it anymore.
    del(arg_high_level)

    # Set up the coordinate system.
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = sp[x1 + arg_low_level]

    # Normalize.
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    # Maximize the length.
    # The ImageJ implementation includes an additional constant when calculating
    # the length, but here we omit it as it does not affect the location of the
    # minimum.
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return sp[arg_level]


def filter_dist_maps(dist_maps, th_mask, min_angles_voxel_count):

    for j in range(dist_maps.shape[3]):
        dist_maps[:,:,:,j] *= th_mask
        dist_label = measure.label(dist_maps[:,:,:,j],connectivity=1)
        dist_props = pd.DataFrame(measure.regionprops_table(dist_label,properties=('label','area')))
        too_small_dist = dist_props.loc[dist_props['area'] < min_angles_voxel_count,'label'].values
        th_mask = np.where(np.isin(dist_label,too_small_dist), 0.0, dist_label)
        th_mask = np.where(th_mask > 0.0, 1.0, 0.0)
        dist_maps[:,:,:,j] *= th_mask

    for j in range(dist_maps.shape[3]):
        dist_maps[:,:,:,j] *= th_mask
    
    return dist_maps, th_mask

def create_angular_distance_maps(angles_map, angles_list, output_file_base = None, write_out_maps = True, c_symmetry = 1 ):

    if output_file_base is None:
        if isinstance(angles_map, str):
            output_file_base = angles_map[:-3]
        elif write_out_maps:         
            ValueError("The output_file_base was not specified -> the maps will not be written out!")
            write_out_maps = False

    angles_map = cryomaps.read(angles_map).astype(int)
    
    map_shape = angles_map.shape
    angles = geom.load_angles(angles_list)
    
    zero_rotations = np.tile(angles[0,:],(angles.shape[0],1))
    dist_all, dist_normals, dist_inplane = geom.compare_rotations(zero_rotations,angles,c_symmetry)
    
    angles_array = angles_map.flatten() - 1
    
    ang_dist_map = dist_all[angles_array].reshape(map_shape)
    dist_normals_map = dist_normals[angles_array].reshape(map_shape)
    dist_inplane_map = dist_inplane[angles_array].reshape(map_shape)

    if write_out_maps:
        cryomaps.write(ang_dist_map , output_file_base + '_dist_all.em', data_type = np.single)
        cryomaps.write(dist_normals_map, output_file_base + '_dist_normals.em', data_type = np.single)
        cryomaps.write(dist_inplane_map, output_file_base + '_dist_inplane.em', data_type = np.single)

    return ang_dist_map, dist_normals_map, dist_inplane_map


def select_peaks(scores_map, angles_map, angles_file, peak_number = None, 
                  create_dist_maps = False, dist_maps_list = ['_dist_all','_dist_normals','_dist_inplane'],
                  dist_maps_name_base = None, write_dist_maps = False, 
                  min_peak_voxel_count = 7, min_angles_voxel_count = 7, template_mask = None, template_radius = 2,
                  edge_masking = None, tomo_mask = None, output_motl_name = None, tomo_number = None
                  ):
    """Automatic peak selection. 

    Args:
        scores_map (ndarray or str): Map with CCC scores (either path to it or loaded as ndarray)
        angles_map (ndarray or str): Map with angle indices (either path to it or loaded as ndarray)
        angles_file (ndarray or str): Angle list used in TM (either path to it or loaded as ndarray). If ndarray is provided it has to be in correct order - phi, theta, psi
        peak_number (int, optional): Number of peaks to return. Defaults to None.
        create_dist_maps (bool, optional): Whether to create distance maps. They have to be provided for the computation. Defaults to False.
        dist_maps_list (list, optional): What distance map(s) to use for the analysis. At least one has to be specified. Defaults to all of them: ['_dist_all','_dist_normals','_dist_inplane'].
        dist_maps_name_base (str, optional): Path and base name of the distance maps. Defaults to None.
        write_dist_maps (bool, optional): Whether to write the created distance maps or not. Used only if create_dist_maps is True. Defaults to False.
        min_peak_voxel_count (int, optional): Size of the minimum volume each peak should have (in voxels). Defaults to 7.
        min_angles_voxel_count (int, optional): Size of the minimum volume each distance map should have around each peak (in voxels). Defaults to 7.
        template_mask (ndarray or str): Mask for masking out the volume around the seleceted peak (either path to it or loaded as ndarray). Ideally a tight mask with hard edges, that is NOT hollow (even for hollow structures). Defaults to None.
        template_radius (int, optional): The radius of a sphere to use for masking out the volume around the selected peak. Used only if the template mask is not specified. Defaults to 2.
        edge_masking (int or ndarray of shape (3,), optional): Dimensions of edges to mask out (). Defaults to None.
        tomo_mask (ndarray, optional): Mask to exclude regions from the analysis. It has to be the same size as the scores map. Defaults to None.
        output_motl_name (str, optional): Name of the output motl. Defaults to None which results in no motl to be written out.
        tomo_number (int, optional): Number of tomogram to be stored in motl. Defaults to None.

    Raises:
        ValueError: If the edge_masking is not specified as one number nor ndsarray of shape (3,)
        ValueError: If the dist_maps_list contains unknown dist map specifier
        ValueError: If the create_dist_maps is False and the dist_maps_name_base is not specified

    Returns:
        output_motl (cryomotl.Motl): Motl with the selected peaks.
        empty_label (ndarray): Map with the selected peaks (same size as scores map).
    """

    # load the angles
    angles = geom.load_angles(angles_file)
    angles_map = (cryomaps.read(angles_map) - 1).astype(int)

    # get threshold and threshold map
    scores_map = cryomaps.read(scores_map)
    th = compute_scores_map_threshold_triangle(scores_map)
    th_map = np.where(scores_map >= th, 1.0, 0.0)

    if tomo_mask is not None:
        th_map *= tomo_mask

    if edge_masking is not None:
        edge_mask = np.zeros(th_map.shape)
        
        if isinstance(edge_masking, int):
            edge_masking = np.full((3,), edge_masking)
        elif edge_masking.shape[0] != 3:
            raise ValueError('The edge mask has to be single number or 3 numbers - one for each dimension.')
        edge_mask[edge_masking[0]:-edge_masking[0], edge_masking[1]:-edge_masking[1], edge_masking[2]:-edge_masking[2]] = 1
        th_map *= edge_mask


    n_dist_maps = len(dist_maps_list)
    dist_maps = np.zeros((th_map.shape[0],th_map.shape[1],th_map.shape[2],n_dist_maps))
    
    if create_dist_maps:
        temp_dist_maps = create_angular_distance_maps(
                                            angles_map, 
                                            angles_file, 
                                            output_file_base = dist_maps_name_base, 
                                            write_out_maps = write_dist_maps )
        for j, d_name in enumerate(dist_maps_list):
            if d_name == "_dist_all":
                dist_maps[:,:,:,j] = temp_dist_maps[0]
            elif d_name == '_dist_normals':
                dist_maps[:,:,:,j] = temp_dist_maps[1]
            elif d_name == '_dist_inplane':
                dist_maps[:,:,:,j] = temp_dist_maps[2]
            else:
                raise ValueError(f'The dist_maps_list contains unknown dist map specifier: {d_name}!')
    elif dist_maps_name_base is None:
        raise ValueError('The dist_maps_name_base was not specified!')
    else:
        for j, d_name in enumerate(dist_maps_list):
            dist_maps[:,:,:,j] = cryomaps.read(dist_maps_name_base+d_name + '.em')

    
    th_map_d = th_map

    labels = measure.label(th_map, connectivity=1)
    props = pd.DataFrame(measure.regionprops_table(labels,properties=('label','area')))
    
    too_small_peaks = props.loc[props['area']<min_peak_voxel_count,'label'].values
    th_map = np.where(np.isin(labels,too_small_peaks),0.0,labels)
    th_map = np.where(th_map > 0.0, 1.0, 0.0)
   
    dist_maps, th_map_d = filter_dist_maps(dist_maps, th_map_d, min_angles_voxel_count)

    for j in range(n_dist_maps):
        dist_temp = np.zeros(th_map.shape)
        dist_label = measure.label(dist_maps[:,:,:,j],connectivity=1)
        dist_props = pd.DataFrame(measure.regionprops_table(dist_label,properties=('label','bbox')))
        labels, xs, xe, ys, ye, zs, ze = dist_props[['label','bbox-0','bbox-3','bbox-1','bbox-4','bbox-2','bbox-5']].T.to_numpy()
        for l in range(labels.shape[0]):
            label_cut = dist_label[xs[l]:xe[l],ys[l]:ye[l],zs[l]:ze[l]]
            label_cut = np.where(label_cut==labels[l],1.0,0.0)
            label_open = morphology.binary_opening(label_cut, footprint=np.ones((2,2,2)), out=None)
            dist_temp[xs[l]:xe[l],ys[l]:ye[l],zs[l]:ze[l]] = np.where(label_open==1, dist_maps[xs[l]:xe[l],ys[l]:ye[l],zs[l]:ze[l],j],0.0)
        dist_maps[:,:,:,j] = dist_temp 

    dist_maps, th_map_d = filter_dist_maps(dist_maps, th_map_d, min_angles_voxel_count)
    
    th_map *= th_map_d

    scores_th = np.ndarray.flatten(scores_map * th_map)
    nz_idx = np.flatnonzero(scores_th)
    remaining_idx = nz_idx[np.argsort(scores_th[nz_idx], axis=None)][::-1]
    selected_peaks = []
    n_selected_peaks = 0

    if template_mask is None:
        particle_mask = cryomask.spherical_mask(2*template_radius+2, radius=template_radius)
    else:
        particle_mask = cryomaps.read(template_mask)

    if peak_number is None:
        peak_number = remaining_idx.size

    c_idx = 0

    empty_label = np.zeros(th_map.shape)
    removed_idx = []

    c_coord = (np.ceil( np.asarray(particle_mask.shape) / 2)).astype(int)

    while n_selected_peaks < peak_number and remaining_idx.size != 0:
        
        idx_3d =  np.unravel_index(remaining_idx[c_idx], th_map.shape)
        ls,le,ms,me = cryomaps.get_start_end_indices(idx_3d, empty_label.shape, particle_mask.shape)        
        cut_coord = c_coord - ms
        
        if template_mask is not None:
            p_particle = cryomaps.rotate(particle_mask, rotation_angles=angles[angles_map[idx_3d[0],idx_3d[1],idx_3d[2]]])
            p_particle = np.where(p_particle>=0.5, 1.0, 0.0)
            p_particle = p_particle[ms[0]:me[0],ms[1]:me[1],ms[2]:me[2]]
        else:
            p_particle = particle_mask[ms[0]:me[0],ms[1]:me[1],ms[2]:me[2]]

        overlap_voxels=np.count_nonzero(empty_label[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2]]*p_particle)

        if overlap_voxels == 0 and np.all(cut_coord<me):
                   
            th_label = measure.label(th_map[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2]]*p_particle)
            th_label_id = th_label[cut_coord[0],cut_coord[1],cut_coord[2]]

            if th_label_id==0:
                peak_area = 0
                angle_size = 0
                print(idx_3d)
            else:
                peak_area = np.count_nonzero(np.where(th_label==th_label_id,1.0,0.0))
                angle_size = min_angles_voxel_count
                for j in range(n_dist_maps):
                    dist_label = measure.label(dist_maps[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2],j]*p_particle)
                    dist_label_id = dist_label[cut_coord[0],cut_coord[1],cut_coord[2]]
                    if dist_label_id == 0:
                        angle_size = 0
                        print(idx_3d)
                        break 
                    else:
                        ## Add opening 
                        label_open = np.where(dist_label==dist_label_id,1.0,0.0)
                        #label_open = morphology.binary_opening(label_open, footprint=np.ones((2,2,2)), out=None)
                        #label_open = measure.label(label_open)
                        #open_label_id = label_open[cut_coord[0],cut_coord[1],cut_coord[2]]
                        #label_open = np.where(label_open==open_label_id,1.0,0.0)
                        #if open_label_id == 0:
                        #    angle_size = 0
                        #    break            
                        angle_size = np.minimum(angle_size,np.count_nonzero(label_open))
                        if angle_size < min_angles_voxel_count:
                            break
            
            if angle_size >= min_angles_voxel_count and peak_area>=min_peak_voxel_count:
                empty_label[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2]] += p_particle
                th_map[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2]] = np.where(p_particle==1,0.0,th_map[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2]])
                for j in range(n_dist_maps):
                    dist_maps[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2],j] = np.where(p_particle==1,0.0,dist_maps[ls[0]:le[0],ls[1]:le[1],ls[2]:le[2],j])
                
                selected_peaks.append((idx_3d, angles[angles_map[idx_3d[0],idx_3d[1],idx_3d[2]]],scores_map[idx_3d[0],idx_3d[1],idx_3d[2]]))
                n_selected_peaks += 1
                non_zero = np.flatnonzero(empty_label)
                remaining_idx = np.setdiff1d(remaining_idx,non_zero,assume_unique=True)
                removed_idx=[]
                c_idx = 0 
            else:     
                removed_idx.append(remaining_idx[c_idx])
                c_idx +=1

        else:
            removed_idx.append(remaining_idx[c_idx])
            c_idx += 1
            
        if c_idx == remaining_idx.size:
            remaining_idx = np.setdiff1d(remaining_idx,np.asarray(removed_idx),assume_unique=True)
            removed_idx=[]
            c_idx = 0 

    motl_df = cryomotl.Motl.create_empty_motl()
    dim, angles, score = zip(*selected_peaks)  
    motl_df[['x','y','z']]= np.array(dim) 
    motl_df[['phi', 'theta', 'psi']] = np.array(angles)
    motl_df['score'] = score
    motl_df=motl_df.fillna(0)
    
    if tomo_number is not None:
        motl_df['tomo_id'] = tomo_number
    
    motl_df['subtomo_id'] = range(1,len(selected_peaks)+1)
    motl_df['class'] = 1

    output_motl = cryomotl.Motl(motl_df)

    if output_motl_name is not None:
        output_motl.write_to_emfile(output_motl_name)

    print(f"Number of selected peaks: {output_motl.df.shape[0]}")

    return output_motl, empty_label