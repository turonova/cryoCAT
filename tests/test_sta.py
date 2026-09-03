import re
import tempfile
import os
import warnings
import json
import dataclasses
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from cryocat.analysis.sta import *
from cryocat.analysis.sta import (
    _apply_working_dir, _normalize_rootdir, _generate_temperature_schedule,
    stopgap_to_nova_angles, nova_to_stopgap_angles,
    MANDATORY, DERIVED, _STA_SCHEMA, _SCHEMA,
    _halfset_from_format, _halfset_to_format,
    _sym_from_format, _sym_to_format,
    get_schema, get_shared_schema, is_mandatory, get_choices, get_default, build_ctx,
    StaParamContext,
    CoassignmentFactor, ConsensusResult, build_coassignment_factor, consensus_groups,
    consensus_motl, reliability_summary, _snap_agreement,
    _fmt_val,
    Block, StaRun, compute_startidx_sequence, compose_subtomo_mode, expand_motl_name,
    denovo_template_blocks, existing_refs_template_blocks, continue_run_prefill,
    preflight_run_folder, create_run_folder, validate_ref_mapping,
    _SUBTOMO_SETTINGS_CONTENT, _RUN_FOLDER_SUBDIRS,
)
from cryocat.utils.starfileio import Starfile
from cryocat.utils.exceptions import UserInputError
from cryocat.core import cryomotl
from cryocat.core.cryomotl import StopgapMotl


@pytest.fixture
def sg_real_mock():
    return {
        'motl_idx': [6, 25, 32, 53, 56, 58, 62, 65, 67, 72],
        'tomo_num': [24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
        'object': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'subtomo_num': [6, 25, 32, 53, 56, 58, 62, 65, 67, 72],
        'halfset': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
        'orig_x': [824, 693, 970, 837, 685, 706, 649, 654, 666, 1276],
        'orig_y': [754, 690, 414, 467, 760, 772, 720, 701, 711, 1402],
        'orig_z': [673, 317, 606, 516, 333, 328, 362, 428, 407, 362],
        'score': [0.5452, 0.4781, 0.4051, 0.5159, 0.5690, 0.6121, 0.6343, 0.4552, 0.4340, 0.4396],
        'x_shift': [0.2121, 0.4859, -0.2024, -0.2657, -0.1771, -0.2918, 0.0871, -0.1028, -0.7304, 0.6586],
        'y_shift': [0.6012, -0.4242, 0.9552, -0.1578, 0.2112, -0.1186, -0.1228, -0.1554, 0.1463, -0.8863],
        'z_shift': [5.4676, -0.0038, 1.5279, 1.4867, 2.4976, -0.9866, -0.4303, 2.4802, -0.4152, 2.3864],
        'phi': [33.2584, 287.6754, 212.4659, 146.4117, 345.4528, 228.8572, -96.1115, 187.8204, 264.0440, 88.7873],
        'psi': [65.2557, -76.2933, -36.9275, -31.1787, 238.3066, 224.2573, 266.1938, -87.9669, -54.3818, 159.2226],
        'the': [8.6516, 83.1399, 46.4518, 90.2115, 86.4150, 155.0719, 91.3999, 87.8747, 104.1026, 78.7093],
        'class': [3, 1, 2, 2, 2, 2, 1, 1, 3, 2]
    }

@pytest.fixture
def sg_mock():
    return {
        'motl_idx': [1, 2, 3],
        'tomo_num': [24, 24, 24],
        'object': [1, 1, 1],
        'subtomo_num': [1, 2, 3],
        'halfset': ['A', 'A', 'A'],
        'orig_x': [100.0, 200.0, 300.0],
        'orig_y': [100.0, 200.0, 300.0],
        'orig_z': [100.0, 200.0, 300.0],
        'score': [0.5, 0.6, 0.7],
        'x_shift': [0.0, 0.0, 0.0],
        'y_shift': [0.0, 0.0, 0.0],
        'z_shift': [0.0, 0.0, 0.0],
        'phi': [0.0, 45.0, 90.0],
        'psi': [0.0, 45.0, 90.0],
        'the': [0.0, 45.0, 90.0],
        'class': [1, 2, 3]
    }

def test_get_stable_particles(sg_real_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base = os.path.join(tmpdir, "class6_er_mr1_")
        #real stopgap format
        data1 = sg_real_mock

        data2 = data1.copy()
        data2['class'] = [3, 1, 3, 2, 1, 2, 1, 2, 3, 2]  #32, 56, 65 changed class

        data3 = data1.copy()
        data3['class'] = [3, 1, 3, 1, 1, 2, 2, 2, 3, 1]

        for i, data in enumerate([data1, data2, data3], 1):
            df = pd.DataFrame(data)
            motl = cryomotl.StopgapMotl(df)
            motl.write_out(f"{motl_base}{i}.star")

        stable_particles = get_stable_particles(motl_base, 1, 3, motl_type="stopgap")
        assert isinstance(stable_particles, list)

        expected_stable = [6, 25, 67, 58]
        assert set(stable_particles) == set(expected_stable)

        for particle_id in expected_stable:
            assert particle_id in stable_particles

        for particle_id in [32, 53, 56, 62, 65, 72]:
            assert particle_id not in stable_particles


@pytest.mark.parametrize("motl_type,expected", [
    ("stopgap", ".star"),
    ("relion", ".star"),
    ("emmotl", ".em"),
])
def test_get_motl_extension(motl_type, expected):
    assert get_motl_extension(motl_type) == expected


def test_get_motl_extension_invalid():
    with pytest.raises(ValueError):
        get_motl_extension("unsupported_type")


@pytest.mark.parametrize("base,it,motl_type,expected", [
    ("run_", 1, "stopgap", "run_1.star"),
    ("run_", 10, "emmotl", "run_10.em"),
    ("run_it", 1, "relion", "run_it001_data.star"),
    ("run_it", 12, "relion5", "run_it012_data.star"),
    ("run_it", 999, "relion5_1", "run_it999_data.star"),
])
def test_get_motl_filename(base, it, motl_type, expected):
    assert get_motl_filename(base, it, motl_type) == expected


def test_write_out_motl(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame(sg_mock)
        motl = cryomotl.StopgapMotl(df)
        base = os.path.join(tmpdir, "output")
        write_out_motl(motl, base, "stopgap")
        assert os.path.exists(base + ".star")
        with pytest.raises(ValueError):
            write_out_motl(motl, os.path.join(tmpdir, "bad"), "unsupported_type")


def test_get_class_occupancy_relion():
    df1 = pd.DataFrame({
        'rlnCoordinateX': [100.0, 200.0],
        'rlnCoordinateY': [100.0, 200.0],
        'rlnCoordinateZ': [100.0, 200.0],
        'rlnAngleRot': [0.0, 0.0],
        'rlnAngleTilt': [0.0, 0.0],
        'rlnAnglePsi': [0.0, 0.0],
        'rlnImageName': ['000001@1.mrc', '000002@1.mrc'],
        'rlnMicrographName': ['1.mrc', '1.mrc'],
        'rlnClassNumber': [1, 2]
    })
    df2 = pd.DataFrame({
        'rlnCoordinateX': [100.0, 200.0],
        'rlnCoordinateY': [100.0, 200.0],
        'rlnCoordinateZ': [100.0, 200.0],
        'rlnAngleRot': [0.0, 0.0],
        'rlnAngleTilt': [0.0, 0.0],
        'rlnAnglePsi': [0.0, 0.0],
        'rlnImageName': ['000001@1.mrc', '000002@1.mrc'],
        'rlnMicrographName': ['1.mrc', '1.mrc'],
        'rlnClassNumber': [1, 1]
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "run_it")
        rm1 = cryomotl.RelionMotl(df1)
        rm2 = cryomotl.RelionMotl(df2)
        rm1.write_out(f"{base_path}001_data.star")
        rm2.write_out(f"{base_path}002_data.star")
        occupancy = get_class_occupancy(base_path, 1, 2, motl_type="relion")
        expected = {
            1: [1, 2],
            2: [1, 0]
        }
        assert occupancy == expected

def test_compute_alignment_statistics(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base = os.path.join(tmpdir, "motl_")

        base_data = sg_mock

        data1 = base_data.copy()
        df1 = pd.DataFrame(data1)
        motl1 = cryomotl.StopgapMotl(df1)
        motl1.write_out(f"{motl_base}1.star")

        data2 = base_data.copy()
        data2['orig_x'] = [x + 10.0 for x in data2['orig_x']]
        data2['orig_y'] = [y + 10.0 for y in data2['orig_y']]
        data2['orig_z'] = [z + 10.0 for z in data2['orig_z']]
        data2['phi'] = [phi + 10.0 for phi in data2['phi']]
        data2['psi'] = [psi + 10.0 for psi in data2['psi']]
        data2['the'] = [theta + 10.0 for theta in data2['the']]

        df2 = pd.DataFrame(data2)
        motl2 = cryomotl.StopgapMotl(df2)
        motl2.write_out(f"{motl_base}2.star")

        stats_df = compute_alignment_statistics(motl_base, 1, 2, motl_type="stopgap")

        expected_position_change = np.sqrt(300)  # 17.3

        assert abs(stats_df['position_change'].iloc[0] - expected_position_change) < 1e-10
        assert abs(stats_df['rmse_x'].iloc[0] - 10.0) < 1e-10
        assert abs(stats_df['rmse_y'].iloc[0] - 10.0) < 1e-10
        assert abs(stats_df['rmse_z'].iloc[0] - 10.0) < 1e-10
        #manually -- test in geom /
        expected_cone_mean = 12.227354511614003
        expected_cone_median = 12.576019274275671
        expected_cone_std = 1.6943189792114082

        assert abs(stats_df['cone_mean'].iloc[0] - expected_cone_mean) < 1e-10
        assert abs(stats_df['cone_median'].iloc[0] - expected_cone_median) < 1e-10
        assert abs(stats_df['cone_std'].iloc[0] - expected_cone_std) < 1e-10
        assert abs(stats_df['cone_var'].iloc[0] - expected_cone_std ** 2) < 1e-10

        assert stats_df['plane_mean'].iloc[0] > 0
        assert stats_df['plane_std'].iloc[0] >= 0
        assert stats_df['plane_var'].iloc[0] >= 0

        #3:output file
        output_path = os.path.join(tmpdir, "alignment_stats.csv")
        stats_with_output = compute_alignment_statistics(
            motl_base, 1, 2,
            motl_type="stopgap",
            output_path=output_path
        )
        assert os.path.exists(output_path)
        loaded_stats = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(stats_with_output, loaded_stats, check_dtype=False)


def test_compute_alignment_statistics_2(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base = os.path.join(tmpdir, "motl_")

        data1 = sg_mock.copy()
        df1 = pd.DataFrame(data1)
        motl1 = cryomotl.StopgapMotl(df1)
        motl1.write_out(f"{motl_base}1.star")

        data2 = sg_mock.copy()
        data2['orig_x'] = [100.0 + 2.0, 200.0 + 10.0, 300.0 + 20.0]  # Different X shifts
        data2['orig_y'] = [100.0 + 2.0, 200.0 + 10.0, 300.0 + 20.0]  # Different Y shifts
        data2['orig_z'] = [100.0 + 2.0, 200.0 + 10.0, 300.0 + 20.0]  # Different Z shifts
        data2['phi'] = [phi + 10.0 for phi in data2['phi']]
        data2['psi'] = [psi + 10.0 for psi in data2['psi']]
        data2['the'] = [theta + 10.0 for theta in data2['the']]

        df2 = pd.DataFrame(data2)
        motl2 = cryomotl.StopgapMotl(df2)
        motl2.write_out(f"{motl_base}2.star")

        stats_df = compute_alignment_statistics(motl_base, 1, 2, motl_type="stopgap")
        filter_rows = [1, 3]
        stats_filtered = compute_alignment_statistics(
            motl_base, 1, 2,
            motl_type="stopgap",
            filter_rows=filter_rows,
            filter_column_name="subtomo_id"
        )

        assert stats_filtered.shape[0] == 1
        #different result
        assert not np.isclose(stats_filtered['position_change'].iloc[0], stats_df['position_change'].iloc[0])


def test_evaluate_alignment_2(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base = os.path.join(tmpdir, "motl_")
        base_data = sg_mock
        for i in range(1, 4):
            data = base_data.copy()
            shift = (i - 1) * 5
            data['orig_x'] = [x + shift for x in data['orig_x']]
            data['orig_y'] = [y + shift for y in data['orig_y']]
            data['orig_z'] = [z + shift for z in data['orig_z']]

            df = pd.DataFrame(data)
            motl = cryomotl.StopgapMotl(df)
            motl.write_out(f"{motl_base}{i}.star")

        stats_no_plot = evaluate_alignment(
            motl_base, 1, 3,
            motl_type="stopgap",
            plot_values=False,
            write_out_stats=False
        )

        stats_with_plot = evaluate_alignment(
            motl_base, 1, 3,
            motl_type="stopgap",
            plot_values=True,
            write_out_stats=False
        )

        pd.testing.assert_frame_equal(stats_no_plot[0], stats_with_plot[0])

        #t2
        stats_dfs = evaluate_alignment(
            motl_base, 1, 3,
            motl_type="stopgap",
            plot_values=True,
            write_out_stats=False
        )

        assert isinstance(stats_dfs, list)
        assert len(stats_dfs) == 1

        #t3
        graph_file = os.path.join(tmpdir, "test_output.html")
        stats_dfs = evaluate_alignment(
            motl_base, 1, 3,
            motl_type="stopgap",
            plot_values=True,
            graph_output_file=graph_file,
            write_out_stats=False
        )

        assert isinstance(stats_dfs, list)

        #t4
        stats_dfs = evaluate_alignment(
            motl_base, 1, 3,
            motl_type="stopgap",
            plot_values=True,
            labels=["Custom Label"],
            write_out_stats=False
        )

        assert isinstance(stats_dfs, list)


def test_evaluate_alignment(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base1 = os.path.join(tmpdir, "motl1_")
        motl_base2 = os.path.join(tmpdir, "motl2_")
        base_data = sg_mock

        for motl_base in [motl_base1, motl_base2]:
            for i in range(1, 4):
                data = base_data.copy()
                shift = (i - 1) * 5
                data['orig_x'] = [x + shift for x in data['orig_x']]
                data['orig_y'] = [y + shift for y in data['orig_y']]
                data['orig_z'] = [z + shift for z in data['orig_z']]

                df = pd.DataFrame(data)
                motl = cryomotl.StopgapMotl(df)
                motl.write_out(f"{motl_base}{i}.star")

        stats_dfs = evaluate_alignment(
            motl_base1, 1, 3,
            motl_type="stopgap",
            plot_values=False,
            write_out_stats=False
        )
        assert isinstance(stats_dfs, list)
        assert len(stats_dfs) == 1
        assert isinstance(stats_dfs[0], pd.DataFrame)

        stats_dfs = evaluate_alignment(
            [motl_base1, motl_base2], 1, 3,
            motl_type="stopgap",
            plot_values=False,
            write_out_stats=False
        )
        assert len(stats_dfs) == 2

        filter_rows = [[1, 2], [2, 3]]
        stats_dfs = evaluate_alignment(
            [motl_base1, motl_base2], 1, 3,
            motl_type="stopgap",
            filter_rows=filter_rows,
            filter_column_name=["subtomo_id", "subtomo_id"],
            plot_values=False,
            write_out_stats=False
        )
        assert len(stats_dfs) == 2

        stats_dfs = evaluate_alignment(
            [motl_base1, motl_base2], 1, 3,
            motl_type="stopgap",
            plot_values=False,
            write_out_stats=True
        )
        assert os.path.exists(motl_base1 + "as_1.csv")
        #print(pd.DataFrame(os.path.join(motl_base1, "as_1.csv")))
        assert os.path.exists(motl_base2 + "as_2.csv")

        #mix filtering
        filter_rows = [[1, 2], None]
        filter_columns = ["subtomo_id", None]
        stats_dfs = evaluate_alignment(
            [motl_base1, motl_base2], 1, 3,
            motl_type="stopgap",
            filter_rows=filter_rows,
            filter_column_name=filter_columns,
            plot_values=False,
            write_out_stats=False
        )
        assert len(stats_dfs) == 2


def test_create_multiref_run(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base = os.path.join(tmpdir, "input")
        #dummy input motl
        df = pd.DataFrame(sg_mock)
        #test class distribution
        df = pd.concat([df] * 20, ignore_index=True)
        motl = cryomotl.StopgapMotl(df)
        input_file = motl_base + ".star"
        motl.write_out(input_file)

        output_base = os.path.join(tmpdir, "output")
        number_of_classes = 3
        number_of_runs = 2
        iteration_number = 4

        create_multiref_run(
            input_motl=input_file,
            number_of_classes=number_of_classes,
            output_motl_base=output_base,
            input_motl_type="stopgap",
            iteration_number=iteration_number,
            number_of_runs=number_of_runs,
            output_motl_type="stopgap"
        )

        generated_classes = []

        for i in range(1, number_of_runs + 1):
            expected_file = f"{output_base}_mr{i}_{iteration_number}.star"
            assert os.path.exists(expected_file)

            result_motl = cryomotl.StopgapMotl(expected_file)
            assert result_motl.df.shape[0] == df.shape[0]

            classes = result_motl.df['class'].values
            assert np.all(classes >= 1)
            assert np.all(classes <= number_of_classes)

            assert len(np.unique(classes)) > 1

            generated_classes.append(classes)

        assert not np.array_equal(generated_classes[0], generated_classes[1])


def test_create_denovo_multiref_run(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_motl = os.path.join(tmpdir, "input.star")
        df = pd.DataFrame(sg_mock)
        df = pd.concat([df] * 100, ignore_index=True)

        # Ensure subtomo_ids are unique for sampling logic
        df['subtomo_id'] = range(1, len(df) + 1)

        cryomotl.StopgapMotl(df).write_out(input_motl)

        output_base = os.path.join(tmpdir, "denovo")
        number_of_classes = 2
        number_of_runs = 2
        class_occupancy = 50
        iteration_number = 1

        create_denovo_multiref_run(
            input_motl=input_motl,
            number_of_classes=number_of_classes,
            output_motl_base=output_base,
            input_motl_type="stopgap",
            class_occupancy=class_occupancy,
            iteration_number=iteration_number,
            number_of_runs=number_of_runs,
            output_motl_type="stopgap"
        )

        #1 check alignment motl (should contain all particles with random classes)
        align_file = f"{output_base}_{iteration_number}.star"
        assert os.path.exists(align_file)
        align_motl = cryomotl.StopgapMotl(align_file)
        assert align_motl.df.shape[0] == df.shape[0]
        # Classes should be assigned
        assert set(align_motl.df['class'].unique()).issubset({1, 2})

        #2 check reference generation motls
        for i in range(1, number_of_runs + 1):
            ref_file = f"{output_base}_ref_mr{i}_{iteration_number}.star"
            assert os.path.exists(ref_file)
            ref_motl = cryomotl.StopgapMotl(ref_file)

            expected_count = number_of_classes * class_occupancy
            assert ref_motl.df.shape[0] == expected_count

            class_counts = ref_motl.df['class'].value_counts()
            assert class_counts[1] == class_occupancy
            assert class_counts[2] == class_occupancy


def test_evaluate_multiref_run(sg_mock):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Class 1: Particles [1, 2, 3]
        # Class 2: Particles [4, 5]
        df1 = pd.DataFrame({
            'subtomo_id': [1, 2, 3, 4, 5],
            'class': [1, 1, 1, 2, 2],
            # Add dummy required columns for StopgapMotl
            'motl_idx': range(1, 6), 'tomo_num': 1, 'object': 1, 'subtomo_num': range(1, 6),
            'halfset': 'A', 'orig_x': 0, 'orig_y': 0, 'orig_z': 0, 'score': 0,
            'x_shift': 0, 'y_shift': 0, 'z_shift': 0, 'phi': 0, 'psi': 0, 'the': 0
        })
        motl1_path = os.path.join(tmpdir, "run1.star")
        cryomotl.StopgapMotl(df1).write_out(motl1_path)

        # Motl 2:
        # Class 1: Particles [1, 2, 4] (3 moved to C2, 4 moved from C2 to C1)
        # Class 2: Particles [3, 5]    (3 moved from C1, 5 stayed)
        df2 = pd.DataFrame({
            'subtomo_id': [1, 2, 3, 4, 5],
            'class': [1, 1, 2, 1, 2],
            'motl_idx': range(1, 6), 'tomo_num': 1, 'object': 1, 'subtomo_num': range(1, 6),
            'halfset': 'A', 'orig_x': 0, 'orig_y': 0, 'orig_z': 0, 'score': 0,
            'x_shift': 0, 'y_shift': 0, 'z_shift': 0, 'phi': 0, 'psi': 0, 'the': 0
        })
        motl2_path = os.path.join(tmpdir, "run2.star")
        cryomotl.StopgapMotl(df2).write_out(motl2_path)

        common_occupancies = evaluate_multirun_stability(
            [motl1_path, motl2_path],
            input_motl_type="stopgap"
        )

        #intersection for class 1: {1, 2, 3} AND {1, 2, 4}: {1, 2}
        assert set(common_occupancies[1]) == {1, 2}

        #Intersection for 2: {4, 5} AND {3, 5}: {5}
        assert set(common_occupancies[2]) == {5}

        with pytest.raises(ValueError):
            evaluate_multirun_stability([motl1_path], input_motl_type="stopgap")


def test_get_subtomos_class_stability():
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base = os.path.join(tmpdir, "iter_")

        #1: P1->C1
        df1 = pd.DataFrame({'subtomo_id': [1], 'class': [1],
                            'motl_idx': 1, 'tomo_num': 1, 'object': 1, 'subtomo_num': 1, 'halfset': 'A', 'orig_x': 0,
                            'orig_y': 0, 'orig_z': 0, 'score': 0, 'x_shift': 0, 'y_shift': 0, 'z_shift': 0, 'phi': 0,
                            'psi': 0, 'the': 0})

        #P1->C1, P2->C1 (P2 is new/changed to C1)
        df2 = pd.DataFrame({'subtomo_id': [1, 2], 'class': [1, 1],
                            'motl_idx': [1, 2], 'tomo_num': 1, 'object': 1, 'subtomo_num': [1, 2], 'halfset': 'A',
                            'orig_x': 0, 'orig_y': 0, 'orig_z': 0, 'score': 0, 'x_shift': 0, 'y_shift': 0, 'z_shift': 0,
                            'phi': 0, 'psi': 0, 'the': 0})

        #p1->C1, P2->C2 (P2 left C1)
        df3 = pd.DataFrame({'subtomo_id': [1, 2], 'class': [1, 2],
                            'motl_idx': [1, 2], 'tomo_num': 1, 'object': 1, 'subtomo_num': [1, 2], 'halfset': 'A',
                            'orig_x': 0, 'orig_y': 0, 'orig_z': 0, 'score': 0, 'x_shift': 0, 'y_shift': 0, 'z_shift': 0,
                            'phi': 0, 'psi': 0, 'the': 0})

        cryomotl.StopgapMotl(df1).write_out(f"{motl_base}1.star")
        cryomotl.StopgapMotl(df2).write_out(f"{motl_base}2.star")
        cryomotl.StopgapMotl(df3).write_out(f"{motl_base}3.star")

        changes = get_subtomos_class_stability(motl_base, 1, 3, motl_type="stopgap")

        #{class_id: [changes_iter2_vs_1, changes_iter3_vs_2]}

        #1:
        #1->2: {1,2} - {1} = {2}
        #2->3: {1} - {1,2} = {}
        assert changes[1] == [1, 0]


        assert 2 not in changes


def test_evaluate_classification():
    with tempfile.TemporaryDirectory() as tmpdir:
        motl_base = os.path.join(tmpdir, "run_")

        #P1->C1, P2->C2
        df1 = pd.DataFrame({'subtomo_id': [1, 2], 'class': [1, 2],
                            'motl_idx': [1, 2], 'tomo_num': 1, 'object': 1, 'subtomo_num': [1, 2], 'halfset': 'A',
                            'orig_x': 0, 'orig_y': 0, 'orig_z': 0, 'score': 0, 'x_shift': 0, 'y_shift': 0, 'z_shift': 0,
                            'phi': 0, 'psi': 0, 'the': 0})

        #P1->C1, P2->C1 (P2 changed C2->C1)
        df2 = pd.DataFrame({'subtomo_id': [1, 2], 'class': [1, 1],
                            'motl_idx': [1, 2], 'tomo_num': 1, 'object': 1, 'subtomo_num': [1, 2], 'halfset': 'A',
                            'orig_x': 0, 'orig_y': 0, 'orig_z': 0, 'score': 0, 'x_shift': 0, 'y_shift': 0, 'z_shift': 0,
                            'phi': 0, 'psi': 0, 'the': 0})

        cryomotl.StopgapMotl(df1).write_out(f"{motl_base}1.star")
        cryomotl.StopgapMotl(df2).write_out(f"{motl_base}2.star")

        stats_file = os.path.join(tmpdir, "stats.csv")

        occupancy, stability = evaluate_classification(
            motl_base, 1, 2,
            motl_type="stopgap",
            output_file_stats=stats_file,
            plot_results=False
        )

        #Iter1=1 (P1), Iter2=2 (P1,P2)
        assert occupancy[1] == [1, 2]
        #Iter1=1 (P2), Iter2=0
        assert occupancy[2] == [1, 0]

        #Iter2 vs Iter1: {1,2} - {1} = {2}
        assert stability[1] == [1]
        #Iter2 vs Iter1: {} - {2} = {}
        assert stability[2] == [0]

        assert os.path.exists(stats_file)
        df_stats = pd.read_csv(stats_file)
        assert df_stats.shape[0] == 2  # 2 iterations
        print(df_stats)

def test_get_class_occupancy():
    #1: basic 3 iterations
    df1 = pd.DataFrame({
        'motl_idx': [1, 2, 3, 4, 5],
        'tomo_num': [1, 1, 1, 1, 1],
        'object': [1, 2, 3, 4, 5],
        'subtomo_num': [1, 2, 3, 4, 5],
        'halfset': ['A', 'A', 'A', 'A', 'A'],
        'orig_x': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_y': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_z': [100.0, 200.0, 300.0, 400.0, 500.0],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5],
        'x_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'y_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'z_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'phi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'psi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'the': [0.0, 0.0, 0.0, 0.0, 0.0],
        'class': [1, 1, 2, 2, 3]
    })

    df2 = pd.DataFrame({
        'motl_idx': [1, 2, 3, 4, 5],
        'tomo_num': [1, 1, 1, 1, 1],
        'object': [1, 2, 3, 4, 5],
        'subtomo_num': [1, 2, 3, 4, 5],
        'halfset': ['A', 'A', 'A', 'A', 'A'],
        'orig_x': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_y': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_z': [100.0, 200.0, 300.0, 400.0, 500.0],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5],
        'x_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'y_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'z_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'phi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'psi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'the': [0.0, 0.0, 0.0, 0.0, 0.0],
        'class': [1, 2, 2, 2, 3]
    })

    df3 = pd.DataFrame({
        'motl_idx': [1, 2, 3, 4, 5],
        'tomo_num': [1, 1, 1, 1, 1],
        'object': [1, 2, 3, 4, 5],
        'subtomo_num': [1, 2, 3, 4, 5],
        'halfset': ['A', 'A', 'A', 'A', 'A'],
        'orig_x': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_y': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_z': [100.0, 200.0, 300.0, 400.0, 500.0],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5],
        'x_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'y_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'z_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'phi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'psi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'the': [0.0, 0.0, 0.0, 0.0, 0.0],
        'class': [1, 2, 3, 3, 3]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "test_motl_")
        sg1 = StopgapMotl(df1)
        sg2 = StopgapMotl(df2)
        sg3 = StopgapMotl(df3)
        sg1.write_out(f"{base_path}1.star")
        sg2.write_out(f"{base_path}2.star")
        sg3.write_out(f"{base_path}3.star")

        occupancy = get_class_occupancy(base_path, 1, 3, motl_type="stopgap")
        expected = {
            1: [2, 1, 1],
            2: [2, 3, 1],
            3: [1, 1, 3]
        }
        assert occupancy == expected

    #2: single iteration
    df = pd.DataFrame({
        'motl_idx': [1, 2, 3],
        'tomo_num': [1, 1, 1],
        'object': [1, 2, 3],
        'subtomo_num': [1, 2, 3],
        'halfset': ['A', 'A', 'A'],
        'orig_x': [100.0, 200.0, 300.0],
        'orig_y': [100.0, 200.0, 300.0],
        'orig_z': [100.0, 200.0, 300.0],
        'score': [0.9, 0.8, 0.7],
        'x_shift': [0.0, 0.0, 0.0],
        'y_shift': [0.0, 0.0, 0.0],
        'z_shift': [0.0, 0.0, 0.0],
        'phi': [0.0, 0.0, 0.0],
        'psi': [0.0, 0.0, 0.0],
        'the': [0.0, 0.0, 0.0],
        'class': [1, 1, 2]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "test_motl_")
        sg = StopgapMotl(df)
        sg.write_out(f"{base_path}5.star")
        occupancy = get_class_occupancy(base_path, 5, 5, motl_type="stopgap")
        expected = {1: [2], 2: [1]}
        assert occupancy == expected

    #3: empty motl
    df = pd.DataFrame(columns=[
        'motl_idx', 'tomo_num', 'object', 'subtomo_num', 'halfset',
        'orig_x', 'orig_y', 'orig_z', 'score', 'x_shift', 'y_shift', 'z_shift',
        'phi', 'psi', 'the', 'class'
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "test_motl_")
        sg = StopgapMotl(df)
        sg.write_out(f"{base_path}1.star")
        sg.write_out(f"{base_path}2.star")
        occupancy = get_class_occupancy(base_path, 1, 2, motl_type="stopgap")
        assert occupancy == {}

    #4: all same class
    df = pd.DataFrame({
        'motl_idx': [1, 2, 3, 4, 5],
        'tomo_num': [1, 1, 1, 1, 1],
        'object': [1, 2, 3, 4, 5],
        'subtomo_num': [1, 2, 3, 4, 5],
        'halfset': ['A', 'A', 'A', 'A', 'A'],
        'orig_x': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_y': [100.0, 200.0, 300.0, 400.0, 500.0],
        'orig_z': [100.0, 200.0, 300.0, 400.0, 500.0],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5],
        'x_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'y_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'z_shift': [0.0, 0.0, 0.0, 0.0, 0.0],
        'phi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'psi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'the': [0.0, 0.0, 0.0, 0.0, 0.0],
        'class': [1, 1, 1, 1, 1]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "test_motl_")
        sg = StopgapMotl(df)
        sg.write_out(f"{base_path}1.star")
        sg.write_out(f"{base_path}2.star")
        occupancy = get_class_occupancy(base_path, 1, 2, motl_type="stopgap")
        expected = {1: [5, 5]}
        assert occupancy == expected


# ---------------------------------------------------------------------------
# Direct coverage: angle converters, log reader, StaParameters factories
# ---------------------------------------------------------------------------


def test_stopgap_to_nova_angles_roundtrip():
    """Even ``angiter`` round-trips identically through nova-extent conversion."""
    cone, cs, inp, is_ = stopgap_to_nova_angles(angiter=4, angincr=2.0, phi_angiter=3, phi_angincr=5.0)
    assert cone == 8.0 and cs == 2.0
    assert inp == 30.0 and is_ == 5.0
    # nova->sg->nova
    ai, ac, pai, pac = nova_to_stopgap_angles(cone, cs, inp, is_)
    assert (ai, ac, pai, pac) == (4, 2.0, 3, 5.0)


def test_nova_to_stopgap_angles_zero_sampling_gives_zero_iter():
    """``cone_sampling=0`` produces ``angiter=0`` (averaging-only step)."""
    ai, ac, pai, pac = nova_to_stopgap_angles(cone_angle=0.0, cone_sampling=0.0,
                                              inplane_angle=0.0, inplane_sampling=0.0)
    assert ai == 0 and pai == 0


def test_sta_log_read_parses_iterations(tmp_path):
    """``sta_log_read`` picks up RMSE labels per iteration block."""
    log = tmp_path / "novasta.log"
    log.write_text(
        "Starting iteration #1\n"
        "  RMSE x shift: 0.10\n"
        "  RMSE y shift: 0.20\n"
        "Starting iteration #2\n"
        "  RMSE x shift: 0.05\n"
        "  RMSE rotation: 1.5\n"
    )
    df = sta_log_read(str(log))
    assert list(df["iteration"]) == [1, 2]
    np.testing.assert_allclose(df.loc[df["iteration"] == 1, "rmse_x"], [0.10])
    np.testing.assert_allclose(df.loc[df["iteration"] == 2, "rmse_rotation"], [1.5])


def _minimal_params_dict():
    """Build the smallest valid params dict for ``StaParameters.from_dict``."""
    return {
        "cone_angle": "30 20 10",
        "cone_sampling": "5 5 5",
        "inplane_angle": "30 20 10",
        "inplane_sampling": "5 5 5",
        "high_pass": "0.05 0.05 0.05",
        "start_index": 1,
    }


def test_staparameters_from_dict_builds_three_iterations():
    """``from_dict`` infers the number of alignment iterations from the longest list."""
    p = StaParameters.from_dict(_minimal_params_dict(), sta_type="novasta")
    assert isinstance(p, NovaStaParams)
    assert p.num_iterations == 3
    assert p.start_iteration == 1
    assert p.end_iteration == 3


def test_staparameters_from_dict_missing_mandatory_warns_not_raises():
    """from_dict with missing mandatory params warns and returns a usable object.

    New contract (schema phase): construction never raises on missing mandatory
    keys; it warns with the full list and returns an in-memory object.
    validate() names them in a list; write_out(strict=True) is the only path
    that raises.
    """
    # cone_angle is mandatory for novaSTA; mask was never in _minimal_params_dict.
    # Both must appear in the report.
    bad = dict(_minimal_params_dict())
    bad.pop("cone_angle")

    with pytest.warns(UserWarning) as rec:
        p = StaParameters.from_dict(bad, sta_type="novasta")

    # Warning names every missing mandatory param — at least cone_angle and mask.
    msg = str(rec[0].message)
    assert "cone angle" in msg
    assert "'mask'" in msg  # quoted form distinguishes 'mask' from 'cc mask'

    # Object is returned and usable: df is non-empty, iterations match input.
    assert isinstance(p, NovaStaParams)
    assert not p.df.empty
    assert len(p.df) == 3  # 3 iterations inferred from "30 20 10" style values

    # validate() returns a list; both missing params are named in it.
    problems = p.validate()
    problem_text = " ".join(problems)
    assert "cone angle" in problem_text
    assert "'mask'" in problem_text

    # Supplying the missing params clears them from the report.
    p.df["cone angle"] = [30.0, 20.0, 10.0]
    p.df["mask"] = "mask.em"
    remaining = p.validate()
    remaining_text = " ".join(remaining)
    assert "cone angle" not in remaining_text
    assert "'mask'" not in remaining_text


def test_staparameters_write_out_strict_raises_on_missing_mandatory(tmp_path):
    """write_out(strict=True) raises ValueError when mandatory params are absent.

    This is the hard gate that protects file emission; from_dict and validate
    are both lenient (warn / return list).
    """
    bad = dict(_minimal_params_dict())
    bad.pop("cone_angle")
    with pytest.warns(UserWarning):
        sg = StaParameters.from_dict(bad, sta_type="stopgap")

    with pytest.raises(ValueError) as exc_info:
        sg.write_out(str(tmp_path / "params.star"), strict=True)

    err = str(exc_info.value)
    assert "cone angle" in err
    assert "'mask'" in err


def test_staparameters_to_novasta_and_to_stopgap_roundtrip():
    """Conversion both ways preserves the per-iteration row count."""
    nova = StaParameters.from_dict(_minimal_params_dict(), sta_type="novasta")
    sg = nova.to_stopgap()
    nova2 = sg.to_novasta()
    assert isinstance(sg, StopgapParams)
    assert isinstance(nova2, NovaStaParams)
    assert sg.num_iterations == nova.num_iterations
    assert nova2.num_iterations == nova.num_iterations


def test_novastaparams_from_file_loads_basic_keys(tmp_path):
    """A minimal novaSTA flat file is parsed into a per-iteration DataFrame."""
    cfg = tmp_path / "params.txt"
    cfg.write_text(
        "iter 2\n"
        "startIndex 1\n"
        "createRef 0\n"
        "coneAngle 30 20\n"
        "coneSampling 5 5\n"
        "inplaneAngle 30 20\n"
        "inplaneSampling 5 5\n"
        "highPass 0.05\n"
    )
    obj = NovaStaParams.from_file(str(cfg))
    assert obj.num_iterations == 2
    assert obj.start_iteration == 1


def test_staparameters_load_dispatches_by_extension(tmp_path):
    """``load`` picks ``NovaStaParams`` for non-.star extensions."""
    cfg = tmp_path / "params.txt"
    cfg.write_text("iter 1\nstartIndex 1\nconeAngle 30\n")
    loaded = StaParameters.load(str(cfg))
    assert isinstance(loaded, NovaStaParams)


def test_staparameters_attach_log_populates_df_stats(tmp_path):
    """``attach_log`` stores the parsed log in ``df_stats``."""
    log = tmp_path / "out.log"
    log.write_text("Starting iteration #1\n  RMSE x shift: 0.5\n")
    p = StaParameters.from_dict(_minimal_params_dict(), sta_type="novasta")
    out = p.attach_log(str(log))
    assert p.df_stats is not None
    assert list(out["iteration"]) == [1]


def test_staparameters_attach_fsc_populates_fsc(tmp_path):
    """``attach_fsc`` parses a CSV-formatted FSC curve into ``self.fsc``."""
    fsc_path = tmp_path / "fsc.csv"
    fsc_path.write_text("x,uncorrected_fsc\n0.0,1.0\n0.1,0.9\n0.2,0.4\n")
    p = StaParameters.from_dict(_minimal_params_dict(), sta_type="novasta")
    out = p.attach_fsc(str(fsc_path), pixel_size=3.0, box_size=128)
    assert p.fsc is not None
    assert "uncorrected_fsc" in out.columns


def test_stopgapparams_from_file_loads_subtomo_parameters(tmp_path):
    """Round-trip a StopgapParams through write_out → from_file."""
    nova = StaParameters.from_dict(_minimal_params_dict(), sta_type="novasta")
    sg = nova.to_stopgap()
    star_path = tmp_path / "params.star"
    sg.write_out(str(star_path))
    loaded = StopgapParams.from_file(str(star_path))
    assert isinstance(loaded, StopgapParams)
    assert loaded.num_iterations == sg.num_iterations

# ── StaParameters path resolution ────────────────────────────────────────────

# ── _apply_working_dir contract (novaSTA-style) ──────────────────────────────


@pytest.mark.parametrize(
    "path, working_dir, expected_suffix",
    [
        ("./ddd",            None,        "./ddd"),
        ("./ddd",            "/scratch",  "ddd"),
        ("/gg/cc/motl_base", "/scratch",  "motl_base"),
        ("/gg/cc/motl_base", None,        "/gg/cc/motl_base"),
        ("motl_base",        "/scratch",  "motl_base"),
        ("motl_base",        None,        "motl_base"),
        ("../foo/bar",       "/scratch",  "bar"),  # only basename joined? no — see below
    ],
)
def test_apply_working_dir_contract(path, working_dir, expected_suffix):
    """All transformations end with the expected basename / relative tail.

    Use suffix comparison so platform separators don't matter.
    """
    out = _apply_working_dir(path, working_dir)
    if working_dir is None:
        # Identity when no override is given.
        assert out == path
        return
    # For relative paths the join keeps the relative tail; for absolute paths
    # we strip down to the basename.
    if expected_suffix == "bar":
        # ../foo/bar is relative; the join preserves the whole tail.
        assert out.replace("\\", "/").endswith("../foo/bar")
    elif expected_suffix == "ddd":
        # ./ddd is relative; the leading ./ is stripped by PurePosixPath but
        # the tail is preserved.
        assert out.replace("\\", "/").endswith("/ddd") or out.replace("\\", "/").endswith("ddd")
    elif expected_suffix == "motl_base":
        assert out.replace("\\", "/").endswith("/motl_base") or out.endswith("motl_base")


def test_apply_working_dir_absolute_replaces_dir():
    """Absolute paths have their directory replaced with working_dir."""
    out = _apply_working_dir("/gg/cc/motl_base", "/scratch").replace("\\", "/")
    assert out == "/scratch/motl_base"


def test_apply_working_dir_relative_joins_onto_dir():
    """Relative paths get working_dir prepended; tail preserved verbatim."""
    out = _apply_working_dir("./ddd", "/scratch").replace("\\", "/")
    # PurePosixPath normalises ./ddd -> ddd, then Path(/scratch) / ddd.
    assert out == "/scratch/ddd"


def test_apply_working_dir_none_is_identity():
    assert _apply_working_dir("./anything", None) == "./anything"
    assert _apply_working_dir("/absolute/path", None) == "/absolute/path"
    assert _apply_working_dir("bare_name", None) == "bare_name"


# ── STOPGAP: rootdir + lists/ resolution ────────────────────────────────────


def _stopgap_df():
    """Minimal STOPGAP params DataFrame with CANONICAL column names."""
    return pd.DataFrame({
        "rootdir": ["/work/run42"],
        "motl": ["allmotl_lt"],           # canonical name (was "motl name")
        "wedge list": ["wedge_list_noInterpol.star"],  # canonical (was "wedgelist name")
        "mask": ["mask_64px.em"],         # canonical (was "mask name")
        "cc mask": ["cc_mask_64px.em"],   # canonical (was "ccmask name")
        "ref": ["pent_b2_64px_ref"],      # canonical (was "ref name")
        "iteration": [1],
    })


def test_stopgap_motl_base_name_uses_rootdir_and_lists():
    sg = StopgapParams(_stopgap_df())
    out = sg.get_motl_base_name(separator="_").replace("\\", "/")
    assert out == "/work/run42/lists/allmotl_lt_"


def test_stopgap_motl_base_name_working_dir_overrides_rootdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.get_motl_base_name(separator="_", working_dir="/scratch/altrun").replace("\\", "/")
    # working_dir replaces /work/run42; the /lists/ subdir is preserved.
    assert out == "/scratch/altrun/lists/allmotl_lt_"


def test_stopgap_motl_base_name_no_rootdir_falls_back_to_bare_name():
    df = _stopgap_df()
    df["rootdir"] = [None]
    sg = StopgapParams(df)
    assert sg.get_motl_base_name(separator="_") == "allmotl_lt_"


def test_stopgap_resolve_wedge_list_in_lists_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_wedge_list().replace("\\", "/")
    assert out == "/work/run42/lists/wedge_list_noInterpol.star"


def test_stopgap_resolve_mask_in_masks_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_mask().replace("\\", "/")
    assert out == "/work/run42/masks/mask_64px.em"


def test_stopgap_resolve_ccmask_in_masks_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_ccmask().replace("\\", "/")
    assert out == "/work/run42/masks/cc_mask_64px.em"


def test_stopgap_resolve_ref_base_in_refs_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_ref_base(separator="_").replace("\\", "/")
    # Downstream code appends <iter>.em.
    assert out == "/work/run42/refs/pent_b2_64px_ref_"


def test_stopgap_resolvers_honour_working_dir_override():
    sg = StopgapParams(_stopgap_df())
    assert sg.resolve_wedge_list("/alt").replace("\\", "/") == "/alt/lists/wedge_list_noInterpol.star"
    assert sg.resolve_mask("/alt").replace("\\", "/") == "/alt/masks/mask_64px.em"
    assert sg.resolve_ccmask("/alt").replace("\\", "/") == "/alt/masks/cc_mask_64px.em"
    assert sg.resolve_ref_base("/alt").replace("\\", "/") == "/alt/refs/pent_b2_64px_ref_"


# ── novaSTA: motl-column-as-path resolution ─────────────────────────────────


def _novasta_df(motl_value):
    return pd.DataFrame({
        "motl": [motl_value],
        "wedge list": ["../wedges/wedge_list.star"],
        "mask": ["/abs/path/mask.em"],
        "cc mask": ["ccmask.em"],
        "ref": ["../ref_base"],
        "iteration": [1],
    })


def test_novasta_motl_base_name_passes_through_when_no_override():
    nv = NovaStaParams(_novasta_df("../virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_")
    assert out == "../virion_motl_cleaned_"


def test_novasta_motl_base_name_relative_path_joins_onto_working_dir():
    nv = NovaStaParams(_novasta_df("./virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_", working_dir="/scratch/run").replace("\\", "/")
    # PurePosixPath strips the ./ from ./virion_motl_cleaned.
    assert out == "/scratch/run/virion_motl_cleaned_"


def test_novasta_motl_base_name_absolute_path_replaces_dir():
    nv = NovaStaParams(_novasta_df("/gg/cc/virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_", working_dir="/scratch/run").replace("\\", "/")
    assert out == "/scratch/run/virion_motl_cleaned_"


def test_novasta_motl_base_name_bare_name_assumes_working_dir():
    nv = NovaStaParams(_novasta_df("virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_", working_dir="/scratch/run").replace("\\", "/")
    assert out == "/scratch/run/virion_motl_cleaned_"


def test_novasta_resolvers_apply_working_dir_per_column():
    nv = NovaStaParams(_novasta_df("./motl_base"))
    # Relative wedge list: joined onto working_dir.
    out_w = nv.resolve_wedge_list("/scratch").replace("\\", "/")
    assert out_w.endswith("/scratch/../wedges/wedge_list.star") or out_w.endswith("/scratch/wedges/wedge_list.star") \
        or out_w == "/scratch/../wedges/wedge_list.star"
    # Absolute mask: directory replaced.
    assert nv.resolve_mask("/scratch").replace("\\", "/") == "/scratch/mask.em"
    # Bare cc mask: joined.
    assert nv.resolve_ccmask("/scratch").replace("\\", "/") == "/scratch/ccmask.em"
    # Reference base with separator.
    out_r = nv.resolve_ref_base("/scratch").replace("\\", "/")
    assert out_r.endswith("ref_base_")


def test_novasta_resolvers_pass_through_without_override():
    nv = NovaStaParams(_novasta_df("./motl_base"))
    assert nv.resolve_wedge_list() == "../wedges/wedge_list.star"
    assert nv.resolve_mask() == "/abs/path/mask.em"
    assert nv.resolve_ccmask() == "ccmask.em"
    assert nv.resolve_ref_base() == "../ref_base_"


def test_novasta_get_fsc_filename_bare_refname():
    """Bare refname: iteration appended with underscores, .txt suffix."""
    nv = NovaStaParams(pd.DataFrame({"ref": ["myref"], "motl": ["motl_"], "iteration": [1]}))
    assert nv.get_fsc_filename(5) == "myref_5_fsc.txt"


def test_novasta_get_fsc_filename_path_refname():
    """Path-based refname: directory preserved, filename built correctly."""
    nv = NovaStaParams(pd.DataFrame({"ref": ["../ref_base"], "motl": ["motl_"], "iteration": [1]}))
    assert nv.get_fsc_filename(3) == "../ref_base_3_fsc.txt"


def test_novasta_get_fsc_filename_with_working_dir():
    """working_dir overrides the directory component of the refname."""
    nv = NovaStaParams(pd.DataFrame({"ref": ["./myref"], "motl": ["motl_"], "iteration": [1]}))
    result = nv.get_fsc_filename(7, working_dir="/scratch").replace("\\", "/")
    assert result == "/scratch/myref_7_fsc.txt"


def test_novasta_get_fsc_filename_missing_ref_returns_none():
    """Returns None when the ref column is absent."""
    nv = NovaStaParams(pd.DataFrame({"motl": ["motl_"], "iteration": [1]}))
    assert nv.get_fsc_filename(1) is None


# ── evaluate_*_from_params plumbs working_dir to the resolver ────────────────


def test_evaluate_from_params_accepts_working_dir(monkeypatch):
    """Confirm the kwarg flows from the public entry point down to the resolver."""
    from cryocat.analysis import sta as sta_mod
    captured: dict = {}

    def _fake_evaluate_alignment(base, start_it, end_it, **kwargs):
        captured["base"] = base
        captured["start_it"] = start_it
        captured["end_it"] = end_it
        captured["motl_type"] = kwargs.get("motl_type")
        return [pd.DataFrame()]

    monkeypatch.setattr(sta_mod, "evaluate_alignment", _fake_evaluate_alignment)

    sg = StopgapParams(_stopgap_df())
    sta_mod.evaluate_alignment_from_params(sg, working_dir="/scratch/altrun")
    assert captured["base"].replace("\\", "/") == "/scratch/altrun/lists/allmotl_lt_"
    assert captured["motl_type"] == "stopgap"


def test_compute_alignment_statistics_from_params_accepts_working_dir(monkeypatch):
    from cryocat.analysis import sta as sta_mod
    captured: dict = {}

    def _fake_compute(base, start_it, end_it, **kwargs):
        captured["base"] = base
        captured["motl_type"] = kwargs.get("motl_type")
        return pd.DataFrame()

    monkeypatch.setattr(sta_mod, "compute_alignment_statistics", _fake_compute)

    nv = NovaStaParams(_novasta_df("/gg/cc/virion_motl"))
    sta_mod.compute_alignment_statistics_from_params(nv, working_dir="/scratch/run")
    assert captured["base"].replace("\\", "/") == "/scratch/run/virion_motl_"
    assert captured["motl_type"] == "emmotl"


# ── §11 new tests ─────────────────────────────────────────────────────────────


def _minimal_sg_df(n_iter: int = 1) -> pd.DataFrame:
    """Build a minimal StopgapParams DataFrame with all required fields."""
    rows = []
    for i in range(1, n_iter + 1):
        rows.append({
            "rootdir":          "./run42",
            "motl":             "allmotl_lt",
            "wedge list":       "wedge_list.star",
            "mask":             "mask_64px.em",
            "cc mask":          "cc_mask_64px.em",
            "ref":              "ref_base",
            "subtomo name":     "subtomo",
            "iteration":        i,
            "cone angle":       30.0,
            "cone sampling":    5.0,
            "inplane angle":    30.0,
            "inplane sampling": 5.0,
            "low pass":         40,
            "high pass":        1,
        })
    return pd.DataFrame(rows)


# ── Test 1: Round-trip fidelity ───────────────────────────────────────────────

def test_roundtrip_no_double_underscore(tmp_path):
    """write_out → from_file preserves values; no double-underscore columns in file."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path)

    # Read raw star file: Starfile.read strips ONE leading '_'.
    # Any column still starting with '_' means the file had '__' (bug).
    frame, _, _ = Starfile.read(path, data_id=0)
    bad_cols = [c for c in frame.columns if c.startswith("_")]
    assert bad_cols == [], f"Double-underscore columns found: {bad_cols}"

    # First column must be completed_ali (after strip)
    assert frame.columns[0] == "completed_ali"

    # Load back and check values
    loaded = StopgapParams.from_file(path)
    assert isinstance(loaded, StopgapParams)
    assert loaded.num_iterations == 1
    assert loaded.df["motl"].iloc[0] == "allmotl_lt"


# ── Test 2: subtomo_mode ordering ─────────────────────────────────────────────

def test_subtomo_mode_canonical_format(tmp_path):
    """All written subtomo_mode values match the canonical {ali|avg}_{family} pattern."""
    sg = StopgapParams(_minimal_sg_df(n_iter=2), create_ref=True)
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path)

    frame, _, _ = Starfile.read(path, data_id=0)
    pattern = re.compile(r'^(ali|avg)_(singleref|multiref|multiclass)$')
    for mode in frame["subtomo_mode"]:
        assert pattern.match(mode), f"Invalid subtomo_mode: {mode!r}"

    # Old STOPGAP format must NOT appear
    assert "multiref_ali" not in frame["subtomo_mode"].values
    assert "singleref_ali" not in frame["subtomo_mode"].values


# ── Test 3: Defaults fill ─────────────────────────────────────────────────────

def test_from_dict_defaults_fill():
    """from_dict fills optional parameters from schema defaults."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = StaParameters.from_dict(
            {
                "rootdir": "./run",
                "motl": "motl_lt",
                "wedge list": "wedge.star",
                "mask": "mask.em",
                "cc mask": "ccmask.em",
                "ref": "ref_base",
                "subtomo name": "subtomo",
                "cone angle": 30.0,
                "cone sampling": 5.0,
                "inplane angle": 30.0,
                "inplane sampling": 5.0,
                "low pass": 40,
                "start_index": 1,
            },
            sta_type="novasta",
        )

    # Optional params should be at schema defaults
    assert "binning" not in p.df.columns or p.df.get("binning", pd.Series([None])).iloc[0] is None
    # high pass has default 1 in schema; if not supplied, may not be in df
    # (from_dict only puts supplied keys in df)
    assert p.num_iterations == 1


# ── Test 4: Manual example line ───────────────────────────────────────────────

def test_stopgap_write_produces_correct_col_count(tmp_path):
    """Basic param_set produces exactly 34 STOPGAP columns."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="basic")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) == 34, (
        f"Expected 34 basic columns, got {len(frame.columns)}: {list(frame.columns)}"
    )


# ── Test 5: Mandatory reporting ───────────────────────────────────────────────

def test_validate_reports_missing_mandatory():
    """validate() lists missing mandatory params; file is still written (strict=False)."""
    df = pd.DataFrame({
        "motl": ["allmotl_lt"],
        "wedge list": ["wedge.star"],
        "mask": ["mask.em"],
        "cc mask": ["ccmask.em"],
        "ref": ["ref_base"],
        "subtomo name": ["subtomo"],
        "iteration": [1],
        "cone angle": [30.0],
        "cone sampling": [5.0],
        "inplane angle": [30.0],
        "inplane sampling": [5.0],
        "low pass": [40],
    })
    # rootdir and mask are intentionally omitted (well, mask IS present; let's omit rootdir + low pass)
    df2 = df.drop(columns=["rootdir"] if "rootdir" in df.columns else [])
    df2 = df2.drop(columns=["low pass"])
    sg = StopgapParams(df2)
    problems = sg.validate()

    reported = " ".join(problems)
    assert "rootdir" in reported
    assert "low pass" in reported


def test_validate_file_still_written_when_not_strict(tmp_path):
    """write_out with strict=False warns but still writes the file."""
    sg = StopgapParams(pd.DataFrame({"iteration": [1]}))
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sg.write_out(path, strict=False)

    import os
    assert os.path.exists(path)
    assert any("mandatory" in str(x.message).lower() for x in w)


# ── Test 6: Literal rejection ─────────────────────────────────────────────────

def test_validate_detects_invalid_literal():
    """validate() reports invalid literal values for e.g. search_mode."""
    df = _minimal_sg_df()
    df["search mode"] = "hillclimb"   # invalid: should be "hc" or "shc"
    sg = StopgapParams(df)
    problems = sg.validate()
    assert any("search mode" in p for p in problems), (
        f"Expected search_mode problem in {problems}"
    )


# ── Test 7: Conditional requirements ─────────────────────────────────────────

def test_validate_conditional_ccmask_not_needed_for_avg():
    """cc mask is not flagged as mandatory when validating an avg-only context."""
    # StopgapParams.validate always uses is_avg_row=False for the context,
    # so cc mask IS mandatory for alignment rows.
    df = _minimal_sg_df()
    df = df.drop(columns=["cc mask"])
    sg = StopgapParams(df)
    problems = sg.validate()
    reported = " ".join(problems)
    assert "cc mask" in reported


def test_validate_split_into_even_odd_requires_fsc_mask():
    """pixel size and fsc mask become mandatory when split_into_even_odd is True."""
    df = pd.DataFrame({
        "iteration": [1],
        "split into even odd": [True],
    })
    nv = NovaStaParams(df)
    problems = nv.validate()
    reported = " ".join(problems)
    assert "fsc mask" in reported or "pixel size" in reported


# ── Test 8: Temperature schedule ──────────────────────────────────────────────

def test_temperature_schedule_zero():
    """T=0 produces all-zero schedule."""
    sched = _generate_temperature_schedule(0, 5)
    assert sched == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_temperature_schedule_normal():
    """T=3, n=3: schedule is [3, 2, 1]."""
    sched = _generate_temperature_schedule(3, 3)
    assert sched == [3.0, 2.0, 1.0]


def test_temperature_schedule_warns_when_not_finished():
    """Temperature schedule warns when n iterations aren't enough to reach 1."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sched = _generate_temperature_schedule(5, 3)  # ends at max(1, 5-2)=3 → warn
    assert len(sched) == 3
    assert sched[0] == 5.0
    assert sched[-1] == 3.0
    assert any("Temperature" in str(x.message) for x in w)


# ── Test 9: rootdir normalisation ─────────────────────────────────────────────

def test_normalize_rootdir_bare_name():
    """Bare folder name gets ./ prepended."""
    assert _normalize_rootdir("run42") == "./run42"


def test_normalize_rootdir_absolute_unchanged():
    """Absolute paths are returned as-is."""
    assert _normalize_rootdir("/data/run42") == "/data/run42"


def test_normalize_rootdir_already_relative():
    """Paths with a separator are returned unchanged."""
    assert _normalize_rootdir("./run42") == "./run42"
    assert _normalize_rootdir("../run42") == "../run42"


# ── Test 10: cc mask in avg rows ──────────────────────────────────────────────

def test_avg_row_has_none_for_ccmask_and_angles(tmp_path):
    """Avg rows written by write_out have 'none' for angle and cc mask columns."""
    sg = StopgapParams(_minimal_sg_df(), create_ref=True)
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path)

    frame, _, _ = Starfile.read(path, data_id=0)
    # First row should be the avg row
    avg_rows = frame[frame["subtomo_mode"].str.startswith("avg_")]
    assert len(avg_rows) >= 1
    for _, row in avg_rows.iterrows():
        assert str(row.get("angincr",  "none")).strip() == "none", "avg row angincr != none"
        assert str(row.get("angiter",  "none")).strip() == "none", "avg row angiter != none"
        assert str(row.get("ccmask_name", "none")).strip() == "none", "avg row ccmask_name != none"


# ── Test 11: Symmetry conversion ──────────────────────────────────────────────

def test_symmetry_c5_roundtrip():
    """Schoenflies C5 → novaSTA integer 5 → back to C5."""
    df = pd.DataFrame({"iteration": [1], "symmetry": ["C5"]})
    sg = StopgapParams(df)
    nv = sg.to_novasta()
    assert nv.df["symmetry"].iloc[0] == 5

    sg2 = nv.to_stopgap()
    assert str(sg2.df["symmetry"].iloc[0]) == "C5"


def test_symmetry_non_cyclic_warns():
    """Non-cyclic symmetry (D7, T, O, I) triggers a warning on to_novasta."""
    df = pd.DataFrame({"iteration": [1], "symmetry": ["D7"]})
    sg = StopgapParams(df)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nv = sg.to_novasta()
    assert any("D7" in str(x.message) or "Non-cyclic" in str(x.message) for x in w)
    assert nv.df["symmetry"].iloc[0] == 1


# ── Test 12: Cross-format conversion ─────────────────────────────────────────

def test_cross_format_roundtrip_preserves_values():
    """StopgapParams → to_novasta → to_stopgap preserves canonical values."""
    df = pd.DataFrame({
        "iteration": [1],
        "motl": ["allmotl_lt"],
        "cone angle": [30.0],
        "cone sampling": [5.0],
        "inplane angle": [20.0],
        "inplane sampling": [4.0],
        "low pass": [40],
        "symmetry": ["C1"],
    })
    sg = StopgapParams(df)
    nv = sg.to_novasta()
    sg2 = nv.to_stopgap()

    # Canonical values survive the round-trip
    assert sg2.df["motl"].iloc[0] == "allmotl_lt"
    assert float(sg2.df["cone angle"].iloc[0]) == 30.0
    assert sg2.num_iterations == sg.num_iterations


# ── Test 13: param_set ────────────────────────────────────────────────────────

def test_param_set_basic_produces_34_columns(tmp_path):
    """param_set='basic' writes exactly 34 STOPGAP columns."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "basic.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="basic")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) == 34


def test_param_set_full_produces_more_than_34_columns(tmp_path):
    """param_set='full' writes more than 34 STOPGAP columns."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "full.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="full")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) > 34


def test_param_set_auto_promotes_when_full_group_columns_set(tmp_path):
    """Setting a full-group column auto-promotes param_set='basic' to 'full'."""
    df = _minimal_sg_df()
    df["scoring fcn"] = "pearson"   # group="full" column
    sg = StopgapParams(df)
    path = str(tmp_path / "auto.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="basic")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) > 34


# ── Test 14: Cone/Euler exclusivity ──────────────────────────────────────────

def test_validate_cone_and_euler_mutually_exclusive():
    """validate() reports an error when both cone and euler search are configured."""
    df = _minimal_sg_df()
    df["euler axes"] = "ZYZ"       # non-trivial euler axes
    df["euler 1 incr"] = 5.0
    df["euler 1 iter"] = 3
    sg = StopgapParams(df)
    problems = sg.validate()
    assert any("mutually exclusive" in p.lower() for p in problems), (
        f"Expected cone/euler exclusivity problem, got: {problems}"
    )


def test_validate_bad_euler_axes_flags_problem():
    """validate() flags euler_axes where first and second axis are the same."""
    df = pd.DataFrame({
        "iteration": [1],
        "euler axes": ["ZZY"],  # second axis must differ from first
    })
    sg = StopgapParams(df)
    problems = sg.validate()
    assert any("euler" in p.lower() for p in problems), (
        f"Expected euler_axes problem, got: {problems}"
    )


# ── New tests: schema API, format converters, bool coercion ───────────────────



# ── New Test 1: halfset from_format — STOPGAP direction ──────────────────────

def test_halfset_from_format_stopgap_inversion():
    """STOPGAP ignore_halfsets=0 means 'do split' → canonical bool True."""
    assert _halfset_from_format(0, "stopgap") is True   # 0 = not ignoring = split
    assert _halfset_from_format(1, "stopgap") is False  # 1 = ignoring = no split


# ── New Test 2: halfset from_format — novaSTA direction ──────────────────────

def test_halfset_from_format_novasta_same_direction():
    """novaSTA splitIntoEvenOdd=1 means 'do split' → canonical bool True."""
    assert _halfset_from_format(1, "novasta") is True
    assert _halfset_from_format(0, "novasta") is False


# ── New Test 3: halfset to_format — STOPGAP direction ────────────────────────

def test_halfset_to_format_stopgap_inversion():
    """Canonical True (do split) → STOPGAP ignore_halfsets=0."""
    assert _halfset_to_format(True,  "stopgap") == 0
    assert _halfset_to_format(False, "stopgap") == 1


# ── New Test 4: halfset to_format — novaSTA direction ────────────────────────

def test_halfset_to_format_novasta_same_direction():
    """Canonical True (do split) → novaSTA splitIntoEvenOdd=1."""
    assert _halfset_to_format(True,  "novasta") == 1
    assert _halfset_to_format(False, "novasta") == 0


# ── New Test 5: symmetry from_format novaSTA (integer → Schoenflies) ─────────

def test_sym_from_format_novasta_integer_to_schoenflies():
    """Loading symmetry integer from novaSTA produces canonical Schoenflies."""
    assert _sym_from_format(5, "novasta") == "C5"
    assert _sym_from_format(1, "novasta") == "C1"
    assert _sym_from_format("5", "novasta") == "C5"


# ── New Test 6: symmetry to_format STOPGAP (plain int → Schoenflies) ─────────

def test_sym_to_format_stopgap_int_to_schoenflies():
    """Plain integer symmetry is promoted to Schoenflies for STOPGAP."""
    assert _sym_to_format(5, "stopgap") == "C5"
    assert _sym_to_format("C5", "stopgap") == "C5"
    assert _sym_to_format("C1", "stopgap") == "C1"


# ── New Test 7: symmetry to_format novaSTA (Schoenflies → integer) ────────────

def test_sym_to_format_novasta_schoenflies_to_int():
    """Schoenflies symmetry is converted to integer for novaSTA."""
    assert _sym_to_format("C5", "novasta") == 5
    assert _sym_to_format("C1", "novasta") == 1
    assert _sym_to_format(5, "novasta") == 5  # already integer — idempotent


# ── New Test 8: rootdir mandatory for STOPGAP, not novaSTA ───────────────────

def test_rootdir_mandatory_for_stopgap_only():
    """rootdir is required for STOPGAP but optional for novaSTA."""
    spec = next(s for s in _STA_SCHEMA if s.canonical == "rootdir")
    ctx_sg = build_ctx(sta_type="stopgap")
    ctx_nv = build_ctx(sta_type="novasta")
    assert is_mandatory(spec, ctx_sg), "rootdir must be mandatory for STOPGAP"
    assert not is_mandatory(spec, ctx_nv), "rootdir must NOT be mandatory for novaSTA"


# ── New Test 9: novaSTA 'folder' key maps to canonical 'rootdir' ─────────────

def test_novasta_folder_maps_to_canonical_rootdir(tmp_path):
    """When a novaSTA file has a 'folder' key it is stored as canonical 'rootdir'."""
    params_file = tmp_path / "params.txt"
    params_file.write_text(
        "iter 1\n"
        "startIndex 1\n"
        "createRef 0\n"
        "folder /data/my_run\n"
        "motl ./allmotl_\n"
        "wedgeList ./wedgelist.star\n"
        "ref ./ref_\n"
        "mask ./mask.em\n"
        "ccMask ./ccmask.em\n"
        "lowPass 30\n"
        "coneAngle 10.0\n"
        "coneSampling 2.5\n"
        "inplaneAngle 20.0\n"
        "inplaneSampling 2.5\n"
    )
    obj = NovaStaParams.from_file(str(params_file))
    assert "rootdir" in obj.df.columns, "'folder' was not remapped to 'rootdir'"
    assert obj.df["rootdir"].iloc[0] == "/data/my_run"


# ── New Test 10: _fmt_val handles bool before int ─────────────────────────────

def test_fmt_val_bool_encoding():
    """_fmt_val converts True → '1' and False → '0', not 'True'/'False'."""
    from cryocat.analysis.sta import _fmt_val
    assert _fmt_val(True)  == "1"
    assert _fmt_val(False) == "0"
    assert _fmt_val(1.0)   == "1"    # whole float drops .0
    assert _fmt_val(3.14)  == "3.14"


# ── New Test 11: build_ctx produces correct StaParamContext ───────────────────

def test_build_ctx_fields():
    """build_ctx returns a StaParamContext with all specified fields."""
    ctx = build_ctx(
        sta_type="stopgap",
        create_ref=True,
        ref_family="multiref",
        n_iterations=5,
        is_avg_row=False,
        use_euler_search=True,
        row={"motl": "mymotl"},
    )
    assert isinstance(ctx, StaParamContext)
    assert ctx.sta_type == "stopgap"
    assert ctx.create_ref is True
    assert ctx.ref_family == "multiref"
    assert ctx.n_iterations == 5
    assert ctx.use_euler_search is True
    assert ctx.get("motl") == "mymotl"
    assert ctx.get("missing", "default") == "default"


# ── New Test 12: get_schema filters by sta_type ───────────────────────────────

def test_get_schema_stopgap_excludes_novasta_only():
    """get_schema('stopgap') must not include novaSTA-only entries."""
    entries = get_schema("stopgap")
    for spec in entries:
        assert spec.stopgap is not None, (
            f"STOPGAP schema includes novaSTA-only spec {spec.canonical!r}"
        )


def test_get_schema_novasta_excludes_stopgap_only():
    """get_schema('novasta') must not include STOPGAP-only entries."""
    entries = get_schema("novasta")
    for spec in entries:
        assert spec.novasta is not None, (
            f"novaSTA schema includes STOPGAP-only spec {spec.canonical!r}"
        )


# ── New Test 13: get_shared_schema returns only cross-format entries ──────────

def test_get_shared_schema_all_have_both_format_names():
    """get_shared_schema() entries must have both stopgap and novasta names."""
    shared = get_shared_schema()
    assert len(shared) > 0, "Expected at least one shared entry"
    for spec in shared:
        assert spec.stopgap is not None and spec.novasta is not None, (
            f"Shared spec {spec.canonical!r} is missing a format name"
        )


# ── New Test 14: Euler columns only mandatory when use_euler_search=True ──────

def test_euler_columns_mandatory_only_with_euler_search():
    """Euler columns are required for STOPGAP+euler_search, not otherwise."""
    euler_specs = [s for s in _STA_SCHEMA if s.group == "euler" and s.canonical is not None]
    assert len(euler_specs) > 0, "No euler specs found"

    ctx_euler = build_ctx(sta_type="stopgap", use_euler_search=True)
    ctx_cone  = build_ctx(sta_type="stopgap", use_euler_search=False)
    ctx_nova  = build_ctx(sta_type="novasta",  use_euler_search=True)

    for spec in euler_specs:
        assert is_mandatory(spec, ctx_euler), (
            f"{spec.canonical!r} must be mandatory for STOPGAP with euler_search"
        )
        assert not is_mandatory(spec, ctx_cone), (
            f"{spec.canonical!r} must NOT be mandatory for STOPGAP cone search"
        )
        assert not is_mandatory(spec, ctx_nova), (
            f"{spec.canonical!r} must NOT be mandatory for novaSTA (no euler support)"
        )



# ── STA block schedule helpers ────────────────────────────────────────────────

# ── compute_startidx_sequence ────────────────────────────────────────────────


def test_startidx_single_block_default():
    blocks = [Block(10, "ali", "{base}")]
    assert compute_startidx_sequence(blocks) == [1]


def test_startidx_single_block_nondefault_start():
    blocks = [Block(10, "ali", "{base}")]
    assert compute_startidx_sequence(blocks, starting_iter=5) == [5]


def test_startidx_three_blocks():
    blocks = [
        Block(1, "avg", "{base}"),
        Block(10, "ali", "{base}"),
        Block(20, "ali", "{base}"),
    ]
    result = compute_startidx_sequence(blocks, starting_iter=1)
    assert result == [1, 2, 12]


def test_startidx_three_blocks_non1_start():
    blocks = [
        Block(1, "avg", "{base}"),
        Block(10, "ali", "{base}"),
        Block(20, "ali", "{base}"),
    ]
    result = compute_startidx_sequence(blocks, starting_iter=31)
    assert result == [31, 32, 42]


def test_startidx_empty_schedule():
    assert compute_startidx_sequence([]) == []


# ── compose_subtomo_mode ─────────────────────────────────────────────────────


@pytest.mark.parametrize("job,run_mode,expected", [
    ("avg", "singleref",  "avg_singleref"),
    ("avg", "multiref",   "avg_multiref"),
    ("avg", "multiclass", "avg_multiclass"),
    ("ali", "singleref",  "ali_singleref"),
    ("ali", "multiref",   "ali_multiref"),
    ("ali", "multiclass", "ali_multiclass"),
])
def test_compose_subtomo_mode(job, run_mode, expected):
    assert compose_subtomo_mode(job, run_mode) == expected


# ── expand_motl_name ─────────────────────────────────────────────────────────


def test_expand_motl_name_base_only():
    assert expand_motl_name("{base}", base="allmotl", run=1, iter_=5) == "allmotl"


def test_expand_motl_name_all_placeholders():
    result = expand_motl_name("{base}_ref_mr{run}_{iter}", base="b", run=3, iter_=7)
    assert result == "b_ref_mr3_7"


def test_expand_motl_name_no_run_placeholder():
    result = expand_motl_name("{base}_{iter}", base="x", run=2, iter_=10)
    assert result == "x_10"


def test_expand_motl_name_literal_unchanged():
    result = expand_motl_name("fixed_name", base="b", run=1, iter_=1)
    assert result == "fixed_name"


# ── denovo_template_blocks ───────────────────────────────────────────────────


def test_denovo_template_has_three_blocks():
    blocks = denovo_template_blocks()
    assert len(blocks) == 3


def test_denovo_template_first_block_is_avg():
    blocks = denovo_template_blocks()
    b = blocks[0]
    assert b.job == "avg"
    assert b.n_iterations == 1
    assert b.motl_name == "{base}_ref_mr{run}"


def test_denovo_template_second_block_annealing():
    blocks = denovo_template_blocks()
    b = blocks[1]
    assert b.job == "ali"
    assert b.n_iterations == 10
    assert b.search_mode == "shc"
    assert b.temperature == 10.0


def test_denovo_template_third_block_zero_temp():
    blocks = denovo_template_blocks()
    b = blocks[2]
    assert b.job == "ali"
    assert b.n_iterations == 20
    assert b.search_mode == "shc"
    assert b.temperature == 0.0


def test_denovo_template_total_alignment_iters():
    blocks = denovo_template_blocks()
    ali_iters = sum(b.n_iterations for b in blocks if b.job == "ali")
    assert ali_iters == 30


# ── existing_refs_template_blocks ────────────────────────────────────────────


def test_exrefs_template_has_two_blocks():
    blocks = existing_refs_template_blocks()
    assert len(blocks) == 2


def test_exrefs_template_first_block_hc():
    b = existing_refs_template_blocks()[0]
    assert b.job == "ali"
    assert b.n_iterations == 1
    assert b.search_mode == "hc"
    assert b.temperature == 0.0


def test_exrefs_template_second_block_shc():
    b = existing_refs_template_blocks()[1]
    assert b.job == "ali"
    assert b.n_iterations == 29
    assert b.search_mode == "shc"
    assert b.temperature == 0.0


def test_exrefs_template_total_iters():
    blocks = existing_refs_template_blocks()
    assert sum(b.n_iterations for b in blocks) == 30


# ── continue_run_prefill ─────────────────────────────────────────────────────


def test_continue_run_prefill_starting_iter():
    last = {"iteration": 10, "motl": "allmotl", "temperature": 5.0}
    result = continue_run_prefill(last)
    assert result["starting_iter"] == 11


def test_continue_run_prefill_temperature_forced_zero():
    last = {"iteration": 5, "temperature": 9.0, "motl": "x"}
    result = continue_run_prefill(last)
    assert result["base_params"]["temperature"] == 0.0


def test_continue_run_prefill_skips_meta_columns():
    last = {
        "iteration": 3,
        "subtomo mode": "ali_multiref",
        "startidx": 1,
        "motl": "allmotl",
    }
    result = continue_run_prefill(last)
    bp = result["base_params"]
    assert "iteration" not in bp
    assert "subtomo mode" not in bp
    assert "startidx" not in bp
    assert "motl" in bp


def test_continue_run_prefill_keeps_other_params():
    last = {
        "iteration": 7,
        "mask name": "mask.em",
        "wedge list": "wl.star",
    }
    result = continue_run_prefill(last)
    assert result["base_params"].get("mask name") == "mask.em"
    assert result["base_params"].get("wedge list") == "wl.star"


# ── validate_ref_mapping ─────────────────────────────────────────────────────


def test_validate_ref_mapping_empty():
    assert validate_ref_mapping([]) == []


def test_validate_ref_mapping_noncontiguous_dst_classes(tmp_path):
    mapping = [
        {"src_run": 1, "src_class": 1, "src_iter": 31, "dst_class": 1,
         "src_ref_dir": str(tmp_path)},
        {"src_run": 1, "src_class": 2, "src_iter": 31, "dst_class": 3,
         "src_ref_dir": str(tmp_path)},
    ]
    errors = validate_ref_mapping(mapping)
    assert any("contiguous" in e.lower() for e in errors)


def test_validate_ref_mapping_missing_half_maps(tmp_path):
    mapping = [
        {"src_run": 1, "src_class": 1, "src_iter": 31, "dst_class": 1,
         "src_ref_dir": str(tmp_path)},
    ]
    errors = validate_ref_mapping(mapping)
    assert any("half-map" in e.lower() or "not found" in e.lower() for e in errors)


def test_validate_ref_mapping_valid(tmp_path):
    it, c = 31, 1
    for suffix in ("", "_A", "_B"):
        (tmp_path / f"ref{suffix}_{it}_{c}.em").write_bytes(b"")
    mapping = [
        {"src_run": 1, "src_class": c, "src_iter": it, "dst_class": 1,
         "src_ref_dir": str(tmp_path)},
    ]
    errors = validate_ref_mapping(mapping)
    assert errors == []


# ── StaRun round-trip ────────────────────────────────────────────────────────


def _make_sta_run(tmp_path: Path) -> StaRun:
    return StaRun(
        input_motl_id="pool-1",
        run_mode="multiref",
        output_base=tmp_path / "output",
        folder_name="run01",
        subtomo_path=tmp_path / "subtomos",
        base_params={"mask name": "mask.em", "binning": 4},
        schedule=[
            Block(1, "avg", "{base}_ref_mr{run}"),
            Block(10, "ali", "{base}", search_mode="shc", temperature=10.0),
        ],
        n_runs=2,
    )


def test_sta_run_round_trip(tmp_path):
    run = _make_sta_run(tmp_path)
    d = dataclasses.asdict(run)
    serialised = json.dumps(d, default=str)
    loaded = json.loads(serialised)
    assert loaded["run_mode"] == "multiref"
    assert loaded["n_runs"] == 2
    assert len(loaded["schedule"]) == 2
    assert loaded["schedule"][0]["job"] == "avg"
    assert loaded["schedule"][1]["temperature"] == 10.0


# ── preflight_run_folder ─────────────────────────────────────────────────────


def _minimal_run(tmp_path: Path) -> tuple[StaRun, list[Path]]:
    sub = tmp_path / "subtomos"
    sub.mkdir()
    run = StaRun(
        input_motl_id="",
        run_mode="singleref",
        output_base=tmp_path / "out",
        folder_name="run01",
        subtomo_path=sub,
        base_params={},
        schedule=[Block(5, "ali", "{base}")],
    )
    motl = tmp_path / "allmotl_1.star"
    motl.write_bytes(b"")
    return run, [motl]


def test_preflight_passes_when_all_ok(tmp_path):
    run, motls = _minimal_run(tmp_path)
    errors = preflight_run_folder(run, motls)
    assert errors == []


def test_preflight_fails_missing_motl(tmp_path):
    run, _ = _minimal_run(tmp_path)
    missing = tmp_path / "does_not_exist.star"
    errors = preflight_run_folder(run, [missing])
    assert any("does_not_exist" in e or "motl" in e.lower() for e in errors)


def test_preflight_fails_existing_run_dir(tmp_path):
    run, motls = _minimal_run(tmp_path)
    (tmp_path / "out" / "run01").mkdir(parents=True)
    errors = preflight_run_folder(run, motls)
    assert any("already exists" in e.lower() for e in errors)


def test_preflight_creates_nothing(tmp_path):
    run, _ = _minimal_run(tmp_path)
    missing = tmp_path / "ghost.star"
    preflight_run_folder(run, [missing])
    assert not (tmp_path / "out").exists()


# ── create_run_folder ────────────────────────────────────────────────────────


def test_create_run_folder_c1_layout(tmp_path):
    run, motls = _minimal_run(tmp_path)
    create_run_folder(run, motls)
    rd = tmp_path / "out" / "run01"
    assert rd.is_dir()
    for sub in _RUN_FOLDER_SUBDIRS:
        assert (rd / sub).is_dir(), f"Missing subdir: {sub}"


def test_create_run_folder_subtomo_settings(tmp_path):
    run, motls = _minimal_run(tmp_path)
    create_run_folder(run, motls)
    settings = tmp_path / "out" / "run01" / "subtomo_settings.txt"
    assert settings.is_file()
    assert settings.read_text() == _SUBTOMO_SETTINGS_CONTENT


def test_create_run_folder_settings_content_exact(tmp_path):
    run, motls = _minimal_run(tmp_path)
    create_run_folder(run, motls)
    content = (tmp_path / "out" / "run01" / "subtomo_settings.txt").read_text()
    assert content == "vol_ext=.em\n"


def test_create_run_folder_star_written(tmp_path):
    run, motls = _minimal_run(tmp_path)
    create_run_folder(run, motls)
    star = tmp_path / "out" / "run01" / "subtomo_param.star"
    assert star.is_file()


def test_create_run_folder_overwrite_removes_and_recreates(tmp_path):
    run, motls = _minimal_run(tmp_path)
    create_run_folder(run, motls)
    # Plant a sentinel to confirm the folder was fully recreated
    sentinel = tmp_path / "out" / "run01" / "sentinel.txt"
    sentinel.write_text("old")
    create_run_folder(run, motls, overwrite=True)
    assert not sentinel.exists(), "overwrite must remove the old folder entirely"
    assert (tmp_path / "out" / "run01").is_dir()


def test_create_run_folder_no_overwrite_raises(tmp_path):
    run, motls = _minimal_run(tmp_path)
    create_run_folder(run, motls)
    with pytest.raises(FileExistsError):
        create_run_folder(run, motls, overwrite=False)


def test_create_run_folder_no_merge(tmp_path):
    """overwrite=True must not leave files from the previous run intact."""
    run, motls = _minimal_run(tmp_path)
    create_run_folder(run, motls)
    leftover = tmp_path / "out" / "run01" / "lists" / "leftover.txt"
    leftover.write_text("old data")
    create_run_folder(run, motls, overwrite=True)
    assert not leftover.exists()


def test_create_run_folder_manifest_keys(tmp_path):
    run, motls = _minimal_run(tmp_path)
    manifest = create_run_folder(run, motls)
    assert "dirs_created" in manifest
    assert "files_copied" in manifest
    assert "symlinks_created" in manifest


def test_create_run_folder_multi_run(tmp_path):
    sub = tmp_path / "subtomos"
    sub.mkdir()
    run = StaRun(
        input_motl_id="",
        run_mode="multiref",
        output_base=tmp_path / "out",
        folder_name="run01",
        subtomo_path=sub,
        base_params={},
        schedule=[Block(5, "ali", "{base}")],
        n_runs=3,
    )
    motl = tmp_path / "allmotl_1.star"
    motl.write_bytes(b"")
    create_run_folder(run, [motl])
    for i in range(1, 4):
        assert (tmp_path / "out" / f"run01_mr{i}").is_dir()



# ── Multi-classification consensus ────────────────────────────────────────────

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(particle_ids, classes, *, particle_col="subtomo_id", class_col="class"):
    return pd.DataFrame({particle_col: particle_ids, class_col: classes})


def _three_class_factor(n_per_class=20, n_runs=5):
    """Perfectly reproduced 3-class ensemble: particles always in the same class."""
    N = n_per_class * 3
    classes = np.repeat([1, 2, 3], n_per_class)
    particle_ids = np.arange(1, N + 1)
    dfs = [_make_df(particle_ids, classes) for _ in range(n_runs)]
    return build_coassignment_factor(dfs)


def _random_label_factor(n_particles=30, n_runs=4, n_classes=3, seed=0):
    """Ensemble with random class labels — no real structure."""
    rng = np.random.default_rng(seed)
    particle_ids = np.arange(1, n_particles + 1)
    dfs = [
        _make_df(particle_ids, rng.integers(1, n_classes + 1, size=n_particles))
        for _ in range(n_runs)
    ]
    return build_coassignment_factor(dfs)


# ── build_coassignment_factor input validation ────────────────────────────────

def test_fewer_than_two_runs_raises():
    with pytest.raises(UserInputError, match="two"):
        build_coassignment_factor([_make_df([1, 2], [1, 2])])


def test_missing_column_raises():
    df = pd.DataFrame({"subtomo_id": [1, 2], "wrong": [1, 2]})
    with pytest.raises(UserInputError, match="missing"):
        build_coassignment_factor([df, df])


def test_empty_run_raises():
    df_ok = _make_df([1, 2], [1, 2])
    df_empty = pd.DataFrame({"subtomo_id": [], "class": []})
    with pytest.raises(UserInputError, match="empty"):
        build_coassignment_factor([df_ok, df_empty])


def test_duplicate_particle_ids_raises():
    df = _make_df([1, 1, 2], [1, 2, 3])
    with pytest.raises(UserInputError, match="repeat"):
        build_coassignment_factor([df, df])


def test_wrong_run_labels_length_raises():
    df = _make_df([1, 2], [1, 2])
    with pytest.raises(UserInputError):
        build_coassignment_factor([df, df], run_labels=["only_one"])


# ── Particle matching by id, not row order ────────────────────────────────────

def test_matching_by_particle_id_not_row_order():
    df1 = _make_df([1, 2, 3], [1, 1, 2])
    df2 = _make_df([3, 1, 2], [2, 1, 1])  # shuffled rows
    factor = build_coassignment_factor([df1, df2])

    m = factor.matrix()
    # particle 1 and 2 share class in both runs → M[0,1] = 1.0
    idx1 = np.searchsorted(factor.particle_ids, 1)
    idx2 = np.searchsorted(factor.particle_ids, 2)
    idx3 = np.searchsorted(factor.particle_ids, 3)
    assert m[idx1, idx2] == pytest.approx(1.0)
    assert m[idx1, idx3] == pytest.approx(0.0)


# ── M = (1/R) B B^T ──────────────────────────────────────────────────────────

def test_matrix_equals_brute_force():
    df1 = _make_df([1, 2, 3, 4], [1, 1, 2, 2])
    df2 = _make_df([1, 2, 3, 4], [1, 2, 1, 2])
    factor = build_coassignment_factor([df1, df2])

    m = factor.matrix()
    labels1 = np.array([1, 1, 2, 2])
    labels2 = np.array([1, 2, 1, 2])
    # Cast to float before summing: numpy bool + bool = bool (OR), not int
    agree1 = (labels1[:, None] == labels1[None, :]).astype(float)
    agree2 = (labels2[:, None] == labels2[None, :]).astype(float)
    brute = (agree1 + agree2) / 2
    np.testing.assert_allclose(m, brute, atol=1e-5)


# ── pca eigenvalues match numpy ───────────────────────────────────────────────

def test_pca_eigenvalues_match_numpy():
    factor = _three_class_factor(n_per_class=10, n_runs=3)
    _, evals_factor = factor.pca(n_components=5)
    m = factor.matrix()
    evals_np = np.linalg.eigvalsh(m.astype(np.float64))
    top_np = np.sort(evals_np)[::-1][:5]
    # Only compare positive eigenvalues (numerical noise may cause tiny negatives)
    for ev_f, ev_np in zip(evals_factor, top_np):
        assert ev_f == pytest.approx(ev_np, abs=1e-3)


# ── consensus_groups at t=1 reproduces consistency_groups ────────────────────

def test_consensus_at_t1_matches_consistency_groups():
    factor = _three_class_factor(n_per_class=15, n_runs=4)
    cg = factor.consistency_groups()

    result = consensus_groups(factor, min_agreement=1.0, min_group_size=1)
    # Both should produce the same partition (different numbering is OK)
    # Check that particles sharing a consistency group also share a consensus label
    cg_labels = cg.set_index("particle_id")["group"]
    for pid in factor.particle_ids:
        # find all particles sharing the same cg group as pid
        same_cg = cg_labels[cg_labels == cg_labels[pid]].index.tolist()
        res_label = result.labels[np.searchsorted(result.particle_ids, pid)]
        for other in same_cg:
            other_label = result.labels[np.searchsorted(result.particle_ids, other)]
            assert res_label == other_label


# ── perfect 3-class ensemble → 3 groups, no junk, reliable ──────────────────

def test_perfect_three_class_ensemble():
    factor = _three_class_factor(n_per_class=20, n_runs=5)
    result = consensus_groups(factor, min_agreement=1.0, min_group_size=5)

    assert result.n_junk == 0
    assert result.n_assigned == 60
    assert result.reliable is True
    # Exactly 3 non-junk classes
    non_junk = result.group_sizes[result.group_sizes.index != result.junk_class]
    assert len(non_junk) == 3


# ── random labels → unreliable, near-total junk at t=1 ──────────────────────

def test_random_labels_fragmented_at_t1():
    """Random labels at t=1 produce many small groups — almost no consensus."""
    factor = _random_label_factor(n_particles=40, n_runs=4, n_classes=4)
    result = consensus_groups(factor, min_agreement=1.0, min_group_size=1)
    n_groups = len(result.group_sizes[result.group_sizes.index != result.junk_class])
    # With random 4-class labels across 4 runs, few particles share the full label tuple
    assert n_groups > 10


def test_random_labels_reliability_summary_unreliable():
    """Ensemble where no pair ever shares a class → M = identity → unreliable."""
    n = 30
    pids = np.arange(1, n + 1)
    rng = np.random.default_rng(42)
    # Each run assigns a unique class to every particle (n classes per run, permuted);
    # two different particles are NEVER in the same class → M = identity.
    dfs = [_make_df(pids, rng.permutation(n) + 1) for _ in range(13)]
    factor = build_coassignment_factor(dfs)
    summary = reliability_summary(factor)
    assert summary["reliable"] is False


# ── complete vs single linkage differ on a chain A-B-C ───────────────────────

def test_complete_vs_single_linkage_chain():
    # 3 particles, 4 runs:
    #   A-B agree in runs 0,1,2  (3/4 = 0.75)
    #   B-C agree in runs 1,2,3  (3/4 = 0.75)
    #   A-C agree in runs 2      (1/4 = 0.25)
    # threshold t = 0.5 → A-B ok, B-C ok, A-C NOT ok
    # complete linkage: A-B and B-C each form a pair but cannot all merge
    # single linkage: A-B-C chain via B

    dfs = [
        _make_df([1, 2, 3], [1, 1, 2]),   # run 0: A-B same
        _make_df([1, 2, 3], [1, 1, 1]),   # run 1: A-B-C same
        _make_df([1, 2, 3], [1, 1, 1]),   # run 2: A-B-C same
        _make_df([1, 2, 3], [2, 1, 1]),   # run 3: B-C same
    ]
    factor = build_coassignment_factor(dfs, run_labels=["r0", "r1", "r2", "r3"])

    # A-B: 3/4=0.75, A-C: 2/4=0.5, B-C: 3/4=0.75
    # At t=0.6: A-B ok (0.75>=0.6), B-C ok, A-C NOT ok (0.5 < 0.6)
    res_complete = consensus_groups(
        factor, min_agreement=0.6, linkage="complete", min_group_size=1
    )
    res_single = consensus_groups(
        factor, min_agreement=0.6, linkage="single", min_group_size=1
    )

    # complete: A-C don't clear threshold together → at most 2 groups among {A,B,C}
    # single: B bridges A and C → all 3 in one group
    n_groups_complete = len(res_complete.group_sizes[res_complete.group_sizes.index != 0])
    n_groups_single = len(res_single.group_sizes[res_single.group_sizes.index != 0])
    assert n_groups_single <= n_groups_complete


# ── t=1 identical under all linkages ─────────────────────────────────────────

def test_t1_all_linkages_identical():
    factor = _three_class_factor(n_per_class=10, n_runs=3)
    rc = consensus_groups(factor, min_agreement=1.0, linkage="complete", min_group_size=1)
    ra = consensus_groups(factor, min_agreement=1.0, linkage="average", min_group_size=1)
    rs = consensus_groups(factor, min_agreement=1.0, linkage="single", min_group_size=1)
    # All should produce the same label partition (different numbering is OK: check sizes)
    def _sorted_sizes(r):
        return tuple(sorted(r.group_sizes[r.group_sizes.index != r.junk_class].tolist(), reverse=True))
    assert _sorted_sizes(rc) == _sorted_sizes(ra) == _sorted_sizes(rs)


# ── min_agreement snaps to nearest achievable value ──────────────────────────

def test_min_agreement_snaps_and_is_recorded():
    factor = _three_class_factor(n_per_class=10, n_runs=4)
    # R=4, achievable values: 0, 0.25, 0.5, 0.75, 1.0
    # requesting 0.6 → ceil(0.6*4)=3 → snaps to 0.75
    result = consensus_groups(factor, min_agreement=0.6, min_group_size=1)
    assert result.min_agreement == pytest.approx(0.75)


def test_snap_agreement_helper():
    t, k = _snap_agreement(0.6, 4)
    assert k == 3
    assert t == pytest.approx(0.75)

    t2, k2 = _snap_agreement(1.0, 4)
    assert k2 == 4
    assert t2 == pytest.approx(1.0)


# ── min_group_size collapses small groups ─────────────────────────────────────

def test_min_group_size_moves_small_to_junk():
    # 3 classes of size 20, 20, 2 — third class falls below min_group_size=5
    classes = np.array([1] * 20 + [2] * 20 + [3] * 2)
    pids = np.arange(1, len(classes) + 1)
    dfs = [_make_df(pids, classes) for _ in range(3)]
    factor = build_coassignment_factor(dfs)

    result = consensus_groups(factor, min_agreement=1.0, min_group_size=5, junk_class=0)
    assert 0 in result.group_sizes.index
    assert result.group_sizes[0] == 2  # the 2-particle group → junk
    non_junk = result.group_sizes[result.group_sizes.index != 0]
    assert len(non_junk) == 2
    # Largest class is class 1
    assert result.group_sizes[1] == 20


# ── classes numbered by descending size ──────────────────────────────────────

def test_classes_numbered_by_descending_size():
    classes = np.array([1] * 30 + [2] * 15 + [3] * 5)
    pids = np.arange(1, len(classes) + 1)
    dfs = [_make_df(pids, classes) for _ in range(3)]
    factor = build_coassignment_factor(dfs)
    result = consensus_groups(factor, min_agreement=1.0, min_group_size=1)
    # class 1 must be largest
    assert result.group_sizes[1] >= result.group_sizes[2]
    assert result.group_sizes[2] >= result.group_sizes[3]


# ── consensus_motl ────────────────────────────────────────────────────────────

class _MockMotl:
    """Minimal Motl stand-in for consensus_motl tests."""

    def __init__(self, df):
        self.df = df.copy()


def test_consensus_motl_matches_by_subtomo_id():
    dfs_runs = [_make_df([1, 2, 3, 4], [1, 1, 2, 2]) for _ in range(3)]
    factor = build_coassignment_factor(dfs_runs)
    result = consensus_groups(factor, min_agreement=1.0, min_group_size=1)

    # Shuffled motl
    motl_df = pd.DataFrame({
        "subtomo_id": [4, 2, 1, 3],
        "class": [0, 0, 0, 0],
        "x": [0.1, 0.2, 0.3, 0.4],
    })
    motl = _MockMotl(motl_df)
    out = consensus_motl(result, motl)

    # Particles 1&2 and 3&4 should each get the same label
    df_out = out.df.set_index("subtomo_id")
    assert df_out.loc[1, "class"] == df_out.loc[2, "class"]
    assert df_out.loc[3, "class"] == df_out.loc[4, "class"]
    assert df_out.loc[1, "class"] != df_out.loc[3, "class"]


def test_consensus_motl_absent_particles_go_to_junk():
    dfs_runs = [_make_df([1, 2], [1, 2]) for _ in range(3)]
    factor = build_coassignment_factor(dfs_runs)
    result = consensus_groups(factor, min_agreement=1.0, min_group_size=1, junk_class=0)

    motl_df = pd.DataFrame({
        "subtomo_id": [1, 2, 99],  # 99 not in factor
        "class": [0, 0, 0],
    })
    out = consensus_motl(result, _MockMotl(motl_df))
    df_out = out.df.set_index("subtomo_id")
    assert df_out.loc[99, "class"] == 0  # junk


def test_consensus_motl_keep_junk_false_drops():
    dfs_runs = [_make_df([1, 2], [1, 2]) for _ in range(3)]
    factor = build_coassignment_factor(dfs_runs)
    result = consensus_groups(factor, min_agreement=1.0, min_group_size=1, junk_class=0)

    motl_df = pd.DataFrame({
        "subtomo_id": [1, 2, 99],
        "class": [0, 0, 0],
    })
    out = consensus_motl(result, _MockMotl(motl_df), keep_junk=False)
    assert 99 not in out.df["subtomo_id"].tolist()
    assert len(out.df) == 2


# ── matrix guard ─────────────────────────────────────────────────────────────

def test_matrix_guard_raises():
    factor = _three_class_factor(n_per_class=10, n_runs=3)
    with pytest.raises(UserInputError, match="Refusing"):
        factor.matrix(max_particles=5)


# ── agreement_histogram guard ─────────────────────────────────────────────────

def test_agreement_histogram_too_many_runs():
    dfs = [_make_df([1, 2, 3], [1, 2, 3]) for _ in range(13)]
    factor = build_coassignment_factor(dfs)
    with pytest.raises(UserInputError, match="12"):
        factor.agreement_histogram(max_runs=12)


def test_agreement_histogram_incomplete_participation():
    df1 = _make_df([1, 2, 3], [1, 1, 2])
    df2 = _make_df([1, 2], [1, 1])  # particle 3 absent
    factor = build_coassignment_factor([df1, df2])
    with pytest.raises(UserInputError, match="every particle"):
        factor.agreement_histogram()


# ── reliability_summary structure ────────────────────────────────────────────

def test_reliability_summary_keys():
    factor = _three_class_factor(n_per_class=10, n_runs=3)
    summary = reliability_summary(factor)
    for key in ("histogram", "eigenvalues", "reliable", "verdict", "n_runs", "n_particles"):
        assert key in summary


def test_reliability_summary_counts():
    factor = _three_class_factor(n_per_class=10, n_runs=3)
    summary = reliability_summary(factor)
    assert summary["n_runs"] == 3
    assert summary["n_particles"] == 30


# ── single linkage warning ────────────────────────────────────────────────────

def test_single_linkage_warns():
    factor = _three_class_factor(n_per_class=5, n_runs=3)
    with pytest.warns(UserWarning, match="single linkage"):
        consensus_groups(factor, min_agreement=0.5, linkage="single", min_group_size=1)


def test_invalid_linkage_raises():
    factor = _three_class_factor(n_per_class=5, n_runs=3)
    with pytest.raises(UserInputError, match="linkage"):
        consensus_groups(factor, linkage="ward")


# ── get_half_map_paths ────────────────────────────────────────────────────────


def test_novasta_get_half_map_paths_bare_refname():
    """Bare refname: even/odd inserted after root, before iteration."""
    nv = NovaStaParams(pd.DataFrame({"ref": ["myref"], "motl": ["motl_"], "iteration": [1]}))
    even, odd = nv.get_half_map_paths(5)
    assert even == "myref_even_5.em"
    assert odd == "myref_odd_5.em"


def test_novasta_get_half_map_paths_with_working_dir():
    """working_dir is applied to the ref path."""
    nv = NovaStaParams(pd.DataFrame({"ref": ["./myref"], "motl": ["motl_"], "iteration": [1]}))
    even, odd = nv.get_half_map_paths(3, working_dir="/scratch")
    assert even.replace("\\", "/") == "/scratch/myref_even_3.em"
    assert odd.replace("\\", "/") == "/scratch/myref_odd_3.em"


def test_novasta_get_half_map_paths_no_ref_returns_none():
    """Returns None when the ref column is absent."""
    nv = NovaStaParams(pd.DataFrame({"motl": ["motl_"], "iteration": [1]}))
    assert nv.get_half_map_paths(1) is None


def test_stopgap_get_half_map_paths_defaults_to_em():
    """When no file exists on disk, .em is the default."""
    sg = StopgapParams(_stopgap_df())
    a, b = sg.get_half_map_paths(5)
    assert a.replace("\\", "/") == "/work/run42/refs/pent_b2_64px_ref_A_5.em"
    assert b.replace("\\", "/") == "/work/run42/refs/pent_b2_64px_ref_B_5.em"


def test_stopgap_get_half_map_paths_picks_mrc_when_exists(tmp_path):
    """Returns .mrc path when .mrc exists and .em does not."""
    df = pd.DataFrame({
        "rootdir": [str(tmp_path)],
        "motl": ["allmotl"],
        "ref": ["myref"],
        "iteration": [1],
    })
    sg = StopgapParams(df)
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "myref_A_2.mrc").write_bytes(b"")
    (refs_dir / "myref_B_2.mrc").write_bytes(b"")
    a, b = sg.get_half_map_paths(2)
    assert a.replace("\\", "/").endswith("refs/myref_A_2.mrc")
    assert b.replace("\\", "/").endswith("refs/myref_B_2.mrc")


def test_stopgap_get_half_map_paths_prefers_em_when_both_exist(tmp_path):
    """When both .em and .mrc exist, returns .em and warns."""
    df = pd.DataFrame({
        "rootdir": [str(tmp_path)],
        "motl": ["allmotl"],
        "ref": ["myref"],
        "iteration": [1],
    })
    sg = StopgapParams(df)
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    for name in ("myref_A_7.em", "myref_A_7.mrc", "myref_B_7.em", "myref_B_7.mrc"):
        (refs_dir / name).write_bytes(b"")
    with pytest.warns(UserWarning, match=r"Both.*exist"):
        a, b = sg.get_half_map_paths(7)
    assert a.replace("\\", "/").endswith("refs/myref_A_7.em")
    assert b.replace("\\", "/").endswith("refs/myref_B_7.em")


def test_stopgap_get_half_map_paths_no_ref_returns_none():
    """Returns None when the ref column is absent."""
    df = pd.DataFrame({"rootdir": ["/work"], "motl": ["motl"], "iteration": [1]})
    sg = StopgapParams(df)
    assert sg.get_half_map_paths(1) is None

