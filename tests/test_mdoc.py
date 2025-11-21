import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import numpy as np
from cryocat import mdoc


class TestMdoc:
    def test_read_mdoc(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        assert hasattr(mdoc_obj, 'titles')
        assert hasattr(mdoc_obj, 'project_info')
        assert hasattr(mdoc_obj, 'imgs')
        assert hasattr(mdoc_obj, 'section_id')
        assert isinstance(mdoc_obj.titles, list)
        assert isinstance(mdoc_obj.project_info, dict)
        assert isinstance(mdoc_obj.imgs, pd.DataFrame)
        assert isinstance(mdoc_obj.section_id, str)
        assert len(mdoc_obj.titles) >= 0
        assert len(mdoc_obj.project_info) > 0
        assert len(mdoc_obj.imgs) > 0

    def test_write_and_read_consistency(self, mdoc_file):
        original_mdoc = mdoc.Mdoc(mdoc_file)
        with tempfile.NamedTemporaryFile(suffix='.mdoc', delete=False, mode='w') as tmp_file:
            tmp_path = tmp_file.name
        try:
            original_mdoc.write(tmp_path, overwrite=True)
            readback_mdoc = mdoc.Mdoc(tmp_path)
            assert original_mdoc.titles == readback_mdoc.titles
            assert original_mdoc.project_info == readback_mdoc.project_info
            assert original_mdoc.section_id == readback_mdoc.section_id
            original_imgs = original_mdoc.imgs.drop(columns=['Removed'], errors='ignore')
            readback_imgs = readback_mdoc.imgs.drop(columns=['Removed'], errors='ignore')
            assert set(original_imgs.columns) == set(readback_imgs.columns)
            for col in original_imgs.columns:
                if col in ['ZValue', 'FrameSet']:
                    assert original_imgs[col].tolist() == readback_imgs[col].tolist()
                elif col in ['TiltAngle', 'PixelSpacing']:
                    assert original_imgs[col].astype(float).tolist() == pytest.approx(
                        readback_imgs[col].astype(float).tolist())
                else:
                    assert original_imgs[col].tolist() == readback_imgs[col].tolist()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_sort_by_tilt(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_tilts = mdoc_obj.imgs['TiltAngle'].tolist()
        mdoc_obj.sort_by_tilt()
        sorted_tilts = mdoc_obj.imgs['TiltAngle'].tolist()
        assert sorted_tilts == sorted(original_tilts)
        assert set(mdoc_obj.imgs.columns) == set(mdoc.Mdoc(mdoc_file).imgs.columns)
        assert len(mdoc_obj.imgs) == len(original_tilts)

    def test_remove_images(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_count = len(mdoc_obj.kept_images())
        num_to_remove = min(2, original_count)
        indices_to_remove = list(range(num_to_remove))
        mdoc_obj.remove_images(indices_to_remove)
        assert len(mdoc_obj.kept_images()) == original_count - num_to_remove
        assert len(mdoc_obj.removed_images()) == num_to_remove
        for idx in indices_to_remove:
            assert mdoc_obj.imgs.loc[idx, 'Removed'] == True

    def test_remove_images_kept_only_false(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_count = len(mdoc_obj.imgs)
        num_to_remove = min(2, original_count)
        indices_to_remove = list(range(num_to_remove))
        mdoc_obj.remove_images(indices_to_remove, kept_only=False)
        assert len(mdoc_obj.kept_images()) == original_count - num_to_remove
        assert len(mdoc_obj.removed_images()) == num_to_remove

    def test_get_image_methods(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        first_image = mdoc_obj.get_image(0)
        assert isinstance(first_image, pd.Series)
        assert len(first_image) > 0
        available_indices = list(range(min(2, len(mdoc_obj.imgs))))
        multiple_images = mdoc_obj.get_images(available_indices)
        assert len(multiple_images) == len(available_indices)
        assert isinstance(multiple_images, pd.DataFrame)
        if 'TiltAngle' in mdoc_obj.imgs.columns and 'ZValue' in mdoc_obj.imgs.columns:
            features = mdoc_obj.get_image_features(['TiltAngle', 'ZValue'])
            assert set(features.columns) == {'TiltAngle', 'ZValue'}
            assert len(features) == len(mdoc_obj.imgs)

    def test_add_field(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_columns = mdoc_obj.imgs.columns.tolist()
        mdoc_obj.add_field("ProcessingNote", "aligned")
        assert "ProcessingNote" in mdoc_obj.imgs.columns
        assert all(mdoc_obj.imgs["ProcessingNote"] == "aligned")
        mdoc_obj.add_field("AlignmentScore", 0.95)
        assert "AlignmentScore" in mdoc_obj.imgs.columns
        assert all(mdoc_obj.imgs["AlignmentScore"] == 0.95)
        for col in original_columns:
            assert col in mdoc_obj.imgs.columns

    def test_update_pixel_size(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        new_pixel_size = 2.5
        mdoc_obj.update_pixel_size(new_pixel_size)
        assert mdoc_obj.project_info["PixelSpacing"] == new_pixel_size
        if "PixelSpacing" in mdoc_obj.imgs.columns:
            assert all(mdoc_obj.imgs["PixelSpacing"] == new_pixel_size)

    def test_write_removed_flag(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        num_images = len(mdoc_obj.imgs)
        indices_to_remove = list(range(min(1, num_images)))
        if indices_to_remove:
            mdoc_obj.remove_images(indices_to_remove)
            with tempfile.NamedTemporaryFile(suffix='.mdoc', delete=False, mode='w') as tmp_file:
                tmp_with_removed = tmp_file.name
            with tempfile.NamedTemporaryFile(suffix='.mdoc', delete=False, mode='w') as tmp_file:
                tmp_without_removed = tmp_file.name
            try:
                mdoc_obj.write(tmp_with_removed, overwrite=True, removed=True)
                mdoc_with_removed = mdoc.Mdoc(tmp_with_removed)
                assert len(mdoc_with_removed.imgs) == len(mdoc_obj.imgs)
                mdoc_obj.write(tmp_without_removed, overwrite=True, removed=False)
                assert os.path.exists(tmp_without_removed)
                with open(tmp_without_removed, 'r') as f:
                    content = f.read()
                assert len(content) > 0
                mdoc_without_removed = mdoc.Mdoc(tmp_without_removed)
            finally:
                for tmp_path in [tmp_with_removed, tmp_without_removed]:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    def test_remove_and_keep_cycle(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_count = len(mdoc_obj.kept_images())
        if original_count > 0:
            mdoc_obj.remove_image(0)
            assert len(mdoc_obj.kept_images()) == original_count - 1
            mdoc_obj.keep_image(0)
            assert len(mdoc_obj.kept_images()) == original_count
            mdoc_obj.reset_images()
            assert len(mdoc_obj.kept_images()) == original_count


class TestMdocWithFiles:
    def test_remove_images(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        num_images = len(mdoc_obj.imgs)
        with tempfile.NamedTemporaryFile(suffix='.mdoc', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            indices_to_remove = list(range(min(1, num_images)))
            if indices_to_remove:
                result = mdoc.remove_images(mdoc_file, indices_to_remove, output_file=tmp_path)
                assert isinstance(result, mdoc.Mdoc)
                assert len(result.removed_images()) == len(indices_to_remove)
                assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_tilt_angles(self, mdoc_file):
        tilt_angles = mdoc.get_tilt_angles(mdoc_file)
        assert isinstance(tilt_angles, (list, pd.Series, np.ndarray))
        assert len(tilt_angles) > 0
        assert all(isinstance(angle, (int, float)) for angle in tilt_angles)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            tilt_angles = mdoc.get_tilt_angles(mdoc_file, output_file=tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path, 'r') as f:
                content = f.read().strip()
            assert len(content) > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_sort_mdoc_by_tilt_angles(self, mdoc_file):
        original_mdoc = mdoc.Mdoc(mdoc_file)
        original_tilts = original_mdoc.imgs['TiltAngle'].tolist()
        result = mdoc.sort_mdoc_by_tilt_angles(mdoc_file, reset_z_value=False)
        sorted_tilts = result.imgs['TiltAngle'].tolist()
        assert sorted_tilts == sorted(original_tilts)
        with tempfile.NamedTemporaryFile(suffix='.mdoc', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            result = mdoc.sort_mdoc_by_tilt_angles(mdoc_file, reset_z_value=True, output_file=tmp_path)
            assert os.path.exists(tmp_path)
            assert result.imgs['ZValue'].tolist() == list(range(len(result.imgs)))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_update_mdoc_features(self, mdoc_file):
        original_mdoc = mdoc.Mdoc(mdoc_file)
        features_to_update = {}
        if 'Voltage' in original_mdoc.project_info:
            features_to_update['Voltage'] = 300
        if 'PixelSpacing' in original_mdoc.project_info:
            features_to_update['PixelSpacing'] = 2.0
        if features_to_update:
            result = mdoc.update_mdoc_features(mdoc_file, features_to_update)
            for feature, value in features_to_update.items():
                assert result.project_info[feature] == value
                if feature in result.imgs.columns:
                    assert all(result.imgs[feature] == value)

    def test_convert_section_type(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_section_id = mdoc_obj.section_id

        target_section_ids = ["FrameSet", "ZValue"] if original_section_id == "ZValue" else ["ZValue", "FrameSet"]

        for target_section_id in target_section_ids:
            if target_section_id != mdoc_obj.section_id:
                mdoc_obj.convert_section_type(target_section_id)
                assert mdoc_obj.section_id == target_section_id
                assert target_section_id in mdoc_obj.imgs.columns
                assert len(mdoc_obj.imgs) > 0
                assert isinstance(mdoc_obj.titles, list)
                assert isinstance(mdoc_obj.project_info, dict)
                assert all(isinstance(col, str) for col in mdoc_obj.imgs.columns)

    def test_merge_mdoc_files(self, mdoc_directory):
        mdoc_files = list(Path(mdoc_directory).glob("*.mdoc"))

        first_file = mdoc_files[0]
        prefix = first_file.stem.split('_')[0]

        merged = mdoc.merge_mdoc_files(
            str(first_file.parent / prefix),
            reorder=True,
            output_file=None
        )

        assert isinstance(merged, mdoc.Mdoc)

        expected_total_images = sum(len(mdoc.Mdoc(str(f)).imgs) for f in mdoc_files)
        assert len(merged.imgs) == expected_total_images

        unique_section_ids = merged.imgs[merged.section_id].unique()
        assert len(unique_section_ids) == len(mdoc_files)
        assert sorted(unique_section_ids) == list(range(len(mdoc_files)))

        tilt_angles = merged.imgs['TiltAngle'].tolist()
        assert tilt_angles == sorted(tilt_angles)

        merged_no_reorder = mdoc.merge_mdoc_files(
            str(first_file.parent / prefix),
            reorder=False,
            output_file=None
        )
        assert len(merged_no_reorder.imgs) == expected_total_images

        with tempfile.NamedTemporaryFile(suffix='.mdoc', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            merged_with_output = mdoc.merge_mdoc_files(
                str(first_file.parent / prefix),
                reorder=True,
                output_file=tmp_path
            )
            assert os.path.exists(tmp_path)
            read_back = mdoc.Mdoc(tmp_path)
            assert len(read_back.imgs) == expected_total_images
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        merged_stripped = mdoc.merge_mdoc_files(
            str(first_file.parent / prefix),
            reorder=True,
            stripFramePath=True,
            output_file=None
        )
        for path in merged_stripped.imgs['SubFramePath']:
            assert path == os.path.basename(path)

        merged_new_id = mdoc.merge_mdoc_files(
            str(first_file.parent / prefix),
            new_id="FrameSet" if merged.section_id == "ZValue" else "ZValue",
            reorder=True,
            output_file=None
        )
        assert merged_new_id.section_id != merged.section_id

    def test_reorder_images(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_order = mdoc_obj.imgs.index.tolist()

        new_order = list(reversed(original_order))
        mdoc_obj.reorder_images(new_order)

        assert mdoc_obj.imgs.index.tolist() == list(range(len(new_order)))

        original_tilt_angles = mdoc_obj.imgs['TiltAngle'].tolist()
        reordered_tilt_angles = list(reversed(original_tilt_angles))
        assert mdoc_obj.imgs['TiltAngle'].tolist() == reordered_tilt_angles

    def test_change_frame_path(self, mdoc_file):
        mdoc_obj = mdoc.Mdoc(mdoc_file)
        original_paths = mdoc_obj.imgs['SubFramePath'].tolist()

        new_path = "/new/path"
        mdoc_obj.change_frame_path(new_path)

        for i, path in enumerate(mdoc_obj.imgs['SubFramePath']):
            expected_path = os.path.join(new_path, os.path.basename(original_paths[i]))
            assert path == expected_path

        mdoc_obj.change_frame_path("")
        for i, path in enumerate(mdoc_obj.imgs['SubFramePath']):
            assert path == os.path.basename(original_paths[i])

    def test_split_mdoc_file(self):
        #todo
        pass

@pytest.fixture
def mdoc_file():
    return str(Path("./test_data/frames_mdoc/TS_019_0000_8.0.mrc.mdoc"))


@pytest.fixture
def mdoc_directory():
    mdoc_path = Path("./test_data/frames_mdoc")
    return str(mdoc_path)