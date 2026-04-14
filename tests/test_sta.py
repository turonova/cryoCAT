import tempfile
import os
import pandas as pd
import pytest
from cryocat.sta import *
from cryocat import *
import pytest
import pandas as pd
import numpy as np
from cryocat import cryomotl
import tempfile
import os

from cryomotl import StopgapMotl


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


def test_get_motl_extension():
    assert get_motl_extension("stopgap") == ".star"
    assert get_motl_extension("relion") == ".star"
    assert get_motl_extension("emmotl") == ".em"

    try:
        get_motl_extension("unsupported_type")
    except ValueError as e:
        assert "unsupported" in str(e).lower()


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
        output_file = os.path.join(tmpdir, "alignment_stats.csv")
        stats_with_output = compute_alignment_statistics(
            motl_base, 1, 2,
            motl_type="stopgap",
            output_file=output_file
        )
        assert os.path.exists(output_file)
        loaded_stats = pd.read_csv(output_file)
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
            filter_column="subtomo_id"
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
        graph_file = os.path.join(tmpdir, "test_output.png")
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
            filter_columns=["subtomo_id", "subtomo_id"],
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
            filter_columns=filter_columns,
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