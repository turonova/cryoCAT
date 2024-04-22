import sys
sys.path.append('.')

gen_dir = './tests/test_data/masks/'

from cryocat.cryomask import *

sm0 = spherical_mask(
    [4, 6, 8],
    2,
    [1, 3, 3],
    0,
    False,
    f"{gen_dir}sm0.em",
)

sm50 = spherical_mask(
    [4, 6, 8],
    2,
    [1, 3, 3],
    0.5,
    False,
    f"{gen_dir}sm50.em",
)

sm50o = spherical_mask(
    [4, 6, 8],
    2,
    [1, 3, 3],
    0.5,
    True,
    f"{gen_dir}sm50o.em",
)

sm50r302010 = rotate(sm50, [0.3, 0.2, 0.1])
write_out(sm50r302010, f"{gen_dir}sm50r302010.em")

cm0 = cylindrical_mask(
    [23, 40, 40],
    7,
    10,
    [10, 10, 15],
    0,
    False,
    None,
    f"{gen_dir}cm0.em"
)

cm25 = cylindrical_mask(
    [23, 40, 40],
    7,
    10,
    [10, 10, 15],
    0.25,
    False,
    None,
    f"{gen_dir}cm25.em"
)
cm25o = cylindrical_mask(
    [23, 40, 40],
    7,
    10,
    [10, 10, 15],
    0.25,
    True,
    None,
    f"{gen_dir}cm25o.em"
)

em0 = ellipsoid_mask(
    [45, 34, 50],
    [5, 10, 15],
    None,
    0,
    f"{gen_dir}em0.em",
    None,
    False
)

em75 = ellipsoid_mask(
    [45, 34, 50],
    [5, 10, 15],
    None,
    0.75,
    f"{gen_dir}em75.em",
    None,
    False
)

em75o = ellipsoid_mask(
    [45, 34, 50],
    [5, 10, 15],
    None,
    0.75,
    f"{gen_dir}em75o.em",
    None,
    True
)
ushape = np.array((25, 50, 35))
um_1 = spherical_mask(
    ushape,
    gaussian=0.7,
    output_name=f"{gen_dir}um_1.em"
)
um_2 = cylindrical_mask(
    ushape,
    7,
    10,
    ushape * 2 // 3,
    gaussian=0.33,
    gaussian_outwards=False,
    angles=np.array((0.31, 0.27, 0.14)),
    output_name=f"{gen_dir}um_2.em"
)
um_3 = ellipsoid_mask(
    ushape,
    np.array((4, 7, 6)),
    ushape // 3,
    angles=np.array((0.5, 0.4, 2)),
    output_name=f"{gen_dir}um_3.em"
)
um = union(
    [um_1, um_2, um_3],
    f"{gen_dir}um.em"
)

mmtmcm25_t10d7 = molmap_tight_mask(
    cm25,
    10,
    7,
    output_name=f"{gen_dir}mmtmcm25_t10d7.em"
)

mtmsm0 = map_tight_mask(
    sm0,
    output_name=f"{gen_dir}mtmsm0.em"
)

smtmsm0_3 = shrink_full_mask(
    mtmsm0,
    3,
    output_name=f"{gen_dir}smtmsm0_3.em"
)

mtmsmtmsm0_3 = map_tight_mask(
    smtmsm0_3,
    output_name=f"{gen_dir}mtmsmtmsm0_3.em"
)

mtmmmtmcm25_t10d7_t40d4g25 = map_tight_mask(
    mmtmcm25_t10d7,
    0.4,
    3,
    0.25,
    output_name=f"{gen_dir}mtmmmtmcm25_t10d7_t40d4g25.em"
)

mtmum_t30d2g20r310040022 = map_tight_mask(
    um,
    0.3,
    2,
    0.2,
    (3.1, 0.4, 2.2),
    output_name=f"{gen_dir}mtmum_t30d2g20r310040022.em"
)

mmtmcm25o_g34F = molmap_tight_mask(
    cm25o,
    gaussian=0.34,
    gaussian_outwards=False,
    output_name=f"{gen_dir}mmtmcm25o_g34F.em"
)

mmtmmtmsm0_t5d2r111222333 = molmap_tight_mask(
    mtmsm0,
    0.05,
    2,
    angles=(1.11, 2.22, 3.33),
    output_name=f"{gen_dir}mmtmmtmsm0_t5d2r111222333.em"
)

sum_4 = shrink_full_mask(
    um, 
    4,
    output_name=f"{gen_dir}sum_4.em"
)
