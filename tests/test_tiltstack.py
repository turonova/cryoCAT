import matplotlib.pyplot as plt
from cryocat.ioutils import fileformat_replace_pattern
from cryocat.tiltstack import *
from pathlib import Path
import pytest
import mrcfile
from scipy.ndimage import gaussian_filter
import os
import shutil


tilt_stack_path = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
tilt_stack_1_path = str(Path(__file__).parent / "test_data" / "tilt_stack_1.mrc")
tilt_stack_2_path = str(Path(__file__).parent / "test_data" / "tilt_stack_2.mrc")
#merging
dir_pattern = str(Path(__file__).parent / "test_data" / "test_merge" / "test_split_*.mrc")
output_pattern = str(Path(__file__).parent /  "test_data" / "test_merge" / "test_merge.mrc")
flipped_path = str(Path(__file__).parent / "test_data" / "flipped_axes.mrc")





def test_crop():
    #case1: new height, new width, default input order, default output order
    cropped_tilt_1 = crop(
        tilt_stack=tilt_stack_path,
        new_width=180,
        new_height=150,
        output_file=tilt_stack_1_path
    )
    #when crop is being called, output_order defines returned np.array shape (in this case it's default)
    assert cropped_tilt_1.shape[0] == 180, "Width of the cropped data is incorrect."
    assert cropped_tilt_1.shape[1] == 150, "Height of the cropped data is incorrect."
    #output file is written in "zyx" format in any case
    ts = TiltStack(tilt_stack=str(tilt_stack_1_path))
    assert ts.data.shape[1] == 150, "Height of the data in the saved file is incorrect."
    assert ts.data.shape[2] == 180, "Width of the data in the saved file is incorrect."
    #case2: use np array somehow instead of reading a file
    cropped_tilt_2 = crop(
        tilt_stack = np.ones((3,3,2), dtype=np.float32))
    assert cropped_tilt_2.shape[0] == 3
    assert cropped_tilt_2.shape[1] == 3
    assert cropped_tilt_2.shape[2] == 2

    """case: pass 2d np array, check if it is 3d returned
    cropped_tilt_3 = crop(
        tilt_stack=np.ones((3, 3), dtype=np.float32))
    assert cropped_tilt_3.shape == 3
    assert cropped_tilt_3.shape[0] == 3
    assert cropped_tilt_3.shape[1] == 3
    assert cropped_tilt_3.shape[2] == 1"""
    #case3: change input order? how without a file?

    #case4: change output order
    cropped_tilt_4 = crop(
        tilt_stack=tilt_stack_path,
        new_width=180,
        new_height=150,
        output_order="zyx")
    assert cropped_tilt_4.shape[0] == 4 #z
    assert cropped_tilt_4.shape[1] == 150 #y
    assert cropped_tilt_4.shape[2] == 180 #x

    #fatal case: what if new width and new height are bigger than original ones?
    ts_original = TiltStack(tilt_stack=tilt_stack_path)
    with pytest.raises(ValueError):
        cropped_tilt_5 = crop(
            tilt_stack = tilt_stack_path,
            new_width=ts_original.width + 10,
            new_height=ts_original.height + 10,
            output_file=tilt_stack_1_path
        )
    #case5: there is no kind of validating data: input could be xyz but my mistake input_order might be zyx, what happens?

    if os.path.exists(tilt_stack_1_path):
        os.remove(tilt_stack_1_path)

def test_sort_tilts_by_angle():
    """with mrcfile.open(tilt_stack_path,permissive=True)as mrc:
        print(mrc.header)"""
    ts = TiltStack(tilt_stack=tilt_stack_path)
    #Supposing that given test file has regularly tilt angles in order, we can try to shuffle 1st and 4th
    tilt_angles = np.array([90.0, 30.0, 60.0, 0.0])
    sorted_stack = sort_tilts_by_angle(
        tilt_stack = tilt_stack_path,
        input_tilts= tilt_angles,
        output_file=tilt_stack_1_path
    )
    print(type(sorted_stack))
    #Visual result confirms the stack is being rearranged considering given iput tilts
    #Let's check it mathematically:
    with mrcfile.open(tilt_stack_path, permissive=True) as mrc_orig:
        original_stack = mrc_orig.data.copy()
    with mrcfile.open(tilt_stack_1_path, permissive=True) as mrc_reord:
        reordered_stack = mrc_reord.data.copy()
    assert np.array_equal(reordered_stack, original_stack[np.argsort(tilt_angles)])



    if os.path.exists(tilt_stack_1_path):
        os.remove(tilt_stack_1_path)

def test_remove_tilts():
    #case1: first tilt to be removed
    #when loading with constructor, order is zyx
    ts_ntilts = TiltStack(tilt_stack_path).data.shape[0]
    ind = [1]
    result = remove_tilts(
        tilt_stack = tilt_stack_path,
        output_file=tilt_stack_1_path,
        idx_to_remove=ind
    )
    #output order is by default xyz
    assert result.shape[2] == ts_ntilts - len(ind)
    #case2: to modify: raise exception when some index is outside bounds of stack
    ind = [1,5] #stack is 4
    with pytest.raises(IndexError):
        result = remove_tilts(
            tilt_stack = tilt_stack_path,
            output_file=tilt_stack_1_path,
            idx_to_remove=ind
        )
    #interesting case: all tilts removed: normal execution with empty stack returned
    ind = [1,2,3,4]
    result = remove_tilts(
        tilt_stack = tilt_stack_path,
        output_file=tilt_stack_2_path,
        idx_to_remove=ind
    )
    assert result.shape[2] == 0 #ts_ntilts - len(ind)
    #interesting case: empty ind list, what happens?
    with pytest.raises(ValueError):
        ind=[]
        result = remove_tilts(
            tilt_stack = tilt_stack_path,
            output_file=tilt_stack_1_path,
            idx_to_remove=ind
        )
        assert result.shape[2] == ts_ntilts
        assert np.equal(result.data, TiltStack(tilt_stack=tilt_stack_path).data)
    if os.path.exists(tilt_stack_1_path):
        os.remove(tilt_stack_1_path)
    if os.path.exists(tilt_stack_2_path):
        os.remove(tilt_stack_2_path)

def test_bin():
    ts=TiltStack(tilt_stack = tilt_stack_path)
    #case1: binning_factor:integer
    binned_stack = bin(
        tilt_stack = tilt_stack_path,
        binning_factor= 2,
        output_file=tilt_stack_1_path
    )
    #returned binned_stack has x,y,z. constructor builds automatically z y x
    assert binned_stack.shape == (ts.data.shape[2]//2, ts.data.shape[1]//2, ts.data.shape[0])
    #assert (TiltStack(tilt_stack_1_path).data == ts.data)
    #case2: binning factor: float. what will happen exception is being raised. ok
    with pytest.raises(Exception):
        binned_stack = bin(
            tilt_stack = tilt_stack_path,
            binning_factor= 0.5,
            output_file=tilt_stack_1_path
        )

def test_calculate_total_dose_batch():
    xmldose_path = str(Path(__file__).parent / "test_data" / "TS_$xxx" / "$xxx.xml")
    tomo_list_path = str(Path(__file__).parent / "test_data" / "tomo_list.txt")
    output_file_path = str(Path(__file__).parent / "test_data" / "TS_$xxx" / "$xxx.txt")
    dose_per_image = 2.0
    calculate_total_dose_batch(
        tomo_list=tomo_list_path,
        prior_dose_file_format=xmldose_path,
        dose_per_image=dose_per_image,
        output_file_format=output_file_path
    )
    for t in ioutils.tlt_load(tomo_list_path).astype(int):
        file_name_output = fileformat_replace_pattern(output_file_path, t, "x", raise_error=False)
        file_name = fileformat_replace_pattern(xmldose_path, t, "x", raise_error=False)
        assert np.allclose(ioutils.total_dose_load(file_name_output), ioutils.total_dose_load(file_name) + dose_per_image)

    #cleanup
    for t in ioutils.tlt_load(tomo_list_path).astype(int):
        if os.path.exists(fileformat_replace_pattern(output_file_path, t, "x", raise_error=False)):
            os.remove(fileformat_replace_pattern(output_file_path, t, "x", raise_error=False))

def test_dose_filter_single_image():
    """
    Since it is not possible to achieve same results of Grant and Grigorieff paper without using Fourier and
    their parameters, it is good practice to test intermediate (basic) functions being used.
    We can test simple cases, e.g. constant image,
    We can visualize the difference between our function's result and Gaussian filter
    """
    tilt_stack = TiltStack(tilt_stack_path)
    # select the first image
    image = tilt_stack.data[0]

    #Example: Gaussian filter plot comparison
    # example parameters
    dose = 5.0  # dose for filtering
    # create a frequency array

    # Use pixel size from your data (e.g., from CTFFind4 output)
    pixel_size = 1.327  # Pixel size in Angstroms
    # Calculate frequency array with proper scaling by pixel size
    freq_x = np.fft.fftfreq(image.shape[1], d=pixel_size)  # d is the pixel size for the x-axis
    freq_y = np.fft.fftfreq(image.shape[0], d=pixel_size)  # d is the pixel size for the y-axis
    # Generate 2D frequency array
    freq_array = np.sqrt(
        np.expand_dims(freq_x, axis=0) ** 2 + np.expand_dims(freq_y, axis=1) ** 2
    )

    # apply the dose filter
    filtered_image = dose_filter_single_image(image, dose, freq_array)
    # calculate expected result using Gaussian filter
    k = 1.0  # scaling factor for sigma
    sigma = k * dose ** 0.5
    expected_result = gaussian_filter(image, sigma=sigma)

    # compare results
    assert filtered_image.shape == expected_result.shape
    assert not np.allclose(image, filtered_image)
    assert not np.allclose(filtered_image, expected_result)

    # plot the original, filtered, and expected images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    axs[1].imshow(filtered_image, cmap='gray')
    axs[1].set_title("Filtered Image (Dose Filter)")
    axs[1].axis('off')
    axs[2].imshow(expected_result, cmap='gray')
    axs[2].set_title("Expected Result (Gaussian Filter)")
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()

    #Test intermediate steps
    #Test fourier inverse
    image = np.random.rand(64, 64)
    ft = np.fft.fft2(image)
    restored_image = np.fft.ifft2(ft).real
    assert np.allclose(image, restored_image, atol=1e-6)
    #Test attenuation
    freq_array = np.linspace(0.1, 10, 100)
    dose = 1.0
    a, b, c = 0.245, -1.665, 2.81
    q = np.exp((-dose) / (2 * ((a * (freq_array**b)) + c)))
    assert np.all(q <= 1.0)
    assert np.all(q >= 0.0)
    """#Test on constant image
    image = np.ones((64, 64))
    freq_array = np.sqrt(
        np.fft.fftfreq(image.shape[0])[:, None] ** 2 + np.fft.fftfreq(image.shape[1])[None, :] ** 2
    )
    dose = 1.0
    filtered_image = dose_filter_single_image(image, dose, freq_array)
    assert np.max(np.abs(image - filtered_image)) < 1e-5, "Filtering constant image failed."""


def test_dose_filter():
    #xml018_path = str(Path(__file__).parent / "test_data" / "TS_018" / "018.xml")
    dose = [8.95372,2.23843,4.47686,6.71529]
    outputfile = str(Path(__file__).parent / "test_data" / "test.mrc")
    pixel_size = 1.327 #from ctffind4 018 tilt..
    #test1, real file mrc
    result = dose_filter(
        tilt_stack = tilt_stack_path,
        pixel_size = pixel_size,
        total_dose=dose,
        output_file=outputfile
    )
    if os.path.exists(outputfile):
        os.remove(outputfile)
    #assert np.allclose(TiltStack(tilt_stack_path).data, TiltStack(outputfile).data)
    #input dose of tilt


def remove1tiltfortesting():
    toberemoved = str(Path(__file__).parent / "test_data" / "test_split_odd.mrc")
    removed = str(Path(__file__).parent / "test_data" / "test_minus1_split.mrc")
    _ = remove_tilts(
        tilt_stack=toberemoved,
        output_file=removed,
        idx_to_remove=[2]
    )
def test_split_stack_even_odd():
    output_file_pattern = str(Path(__file__).parent / "test_data" / "test_split")
    even, odd = split_stack_even_odd(
        tilt_stack=tilt_stack_path,
        output_file_prefix=output_file_pattern
    )
    # Load the original tilt stack
    ts = TiltStack(tilt_stack_path)
    tscopy = ts.data.transpose(2,1,0)

   # Compare the even and odd stacks with the original data
    assert np.allclose(tscopy[:,:,0],even[:,:,0])
    assert np.allclose(tscopy[:,:,1],odd[:,:,0])
    assert np.allclose(tscopy[:,:,2],even[:,:,1])
    assert np.allclose(tscopy[:,:,3],odd[:,:,1])

    #odd number of tilts, what happens?
    #just 1 tilt, what happens?
    with pytest.raises(ValueError):
        remove1tiltfortesting()
        just1tilt = str(Path(__file__).parent / "test_data" / "test_minus1_split.mrc")
        output_file_pattern = str(Path(__file__).parent / "test_data" / "test2_split")
        even, odd = split_stack_even_odd(
            tilt_stack=just1tilt,
            output_file_prefix=output_file_pattern
        )

def create_files_for_merging_test():
    src = str(Path(__file__).parent / "test_data" / "test_split_odd.mrc")
    dst_directory = str(Path(__file__).parent / "test_data" / "test_merge")
    new_filename = 'test_split_1.mrc'
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
    dst = os.path.join(dst_directory, new_filename)
    shutil.copy(src, dst)

    src = str(Path(__file__).parent / "test_data" / "test_split_even.mrc")
    new_filename = 'test_split_2.mrc'
    dst = os.path.join(dst_directory, new_filename)
    shutil.copy(src, dst)

def test_merge():
    create_files_for_merging_test()
    res = merge(
        file_path_pattern=dir_pattern,
        output_file=output_pattern
    )
    ts1path = str(Path(__file__).parent /  "test_data" / "test_merge" / "test_split_1.mrc")
    ts1 = TiltStack(ts1path)
    ts2path = str(Path(__file__).parent / "test_data" / "test_merge" / "test_split_2.mrc")
    ts2 = TiltStack(ts2path)
    ts1c = ts1.data.transpose(2,1,0)
    ts2c = ts2.data.transpose(2,1,0)
    for i in range(res.shape[2]):
        if i<res.shape[2]/2 :
            assert np.allclose(ts1c[:,:,i],res[:,:,i])
        else :
            assert np.allclose(ts2c[:,:,i-res.shape[2]/2],res[:,:,i])

def test_flip_along_axes():
    flipped = flip_along_axes(
        tilt_stack = tilt_stack_path,
        output_file=flipped_path,
        axes = ["x","y","z"]
    )
    ts = TiltStack(tilt_stack_path)
    tsc = ts.data.transpose(2,1,0)
    assert np.allclose(tsc[::-1, ::-1, ::-1],flipped[:,:,:])

def test_deconvolve():
    """
    The deconvolve function is designed to improve the quality of a tilt series (MRC file or NumPy array)
    by applying CTF correction and Wiener deconvolution. This is commonly used in Cryo-EM tomography to
    enhance image contrast and remove distortions caused by the microscope optics.
    """
    tilt_stack_path = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    expected_output_path = str(Path(__file__).parent / "test_data" / "expected_output.mrc")
    synthetic_input = np.random.rand(4, 128, 128).astype(np.float32)
    output_mrc = deconvolve(tilt_stack_path, pixel_size_a=3.42, defocus=2.5, output_file=expected_output_path)
    expected_mrc = TiltStack(expected_output_path).data.transpose(2,1,0)
    assert np.allclose(output_mrc, expected_mrc, atol=1e-5)
    output_numpy = deconvolve(synthetic_input, pixel_size_a=3.42, defocus=4)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(synthetic_input[0], cmap="gray")
    plt.title("Original Tilt Image")
    plt.subplot(1, 2, 2)
    plt.imshow(output_numpy[0], cmap="gray")
    plt.title("Deconvolved Tilt Image")
    plt.show()

    assert output_numpy.shape == synthetic_input.shape
    assert np.isfinite(output_numpy).all()
    assert not np.allclose(output_numpy, synthetic_input)
    assert output_numpy.dtype == synthetic_input.dtype
    assert np.max(output_numpy) <= np.max(synthetic_input) * 2

    input_fft = np.abs(np.fft.fftshift(np.fft.fftn(synthetic_input)))
    output_fft = np.abs(np.fft.fftshift(np.fft.fftn(output_numpy)))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log1p(np.mean(input_fft, axis=0)), cmap="inferno")
    plt.title("FFT Input")
    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(np.mean(output_fft, axis=0)), cmap="inferno")
    plt.title("FFT Output (Deconvolved)")
    plt.show()

def test_equalize_histogram():
    def plot_histograms(original, processed, title):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(original.ravel(), bins=256, color='blue', alpha=0.6, label='Original')
        ax[1].hist(processed.ravel(), bins=256, color='red', alpha=0.6, label='Processed')
        ax[0].set_title("Original Histogram")
        ax[1].set_title(title)
        ax[0].legend()
        ax[1].legend()
        plt.show()

    def check_statistics(original, processed):
        print("Original Min:", original.min(), "Processed Min:", processed.min())
        print("Original Max:", original.max(), "Processed Max:", processed.max())
        print("Original Mean:", original.mean(), "Processed Mean:", processed.mean())
        print("Original Std:", original.std(), "Processed Std:", processed.std())

    with pytest.raises(ValueError):
        equalize_histogram(
            tilt_stack=tilt_stack_path,
            output_file=tilt_stack_1_path,
            eh_method="notvalid"
        )
    synthetic_input = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    synthetic_data = TiltStack(synthetic_input).data
    for method in ["contrast_stretching", "equalization", "adaptive_eq"]:
        output_data = equalize_histogram(synthetic_input, eh_method=method)
        output_data = TiltStack(output_data).data
        assert output_data.shape == synthetic_data.shape, "Output shape mismatch"
        assert output_data.dtype == synthetic_data.dtype, "Output data type mismatch"
        assert np.isfinite(output_data).all(), "Output contains NaN or Inf values"
        assert not np.allclose(output_data, synthetic_data), "Output is too similar to input"

        plot_histograms(synthetic_data[0], output_data[0], f"{method} Histogram")
        check_statistics(synthetic_data[0], output_data[0])

        if method == "equalization":
            hist_input, _ = np.histogram(synthetic_data.flatten(), bins=256)
            hist_output, _ = np.histogram(output_data.flatten(), bins=256)
            assert np.std(hist_output) < np.std(hist_input), "Histogram should be more uniform"

        if method == "contrast_stretching":
            assert np.min(output_data.astype(np.float64)) == 0, "Contrast stretching failed (min)"
            assert np.max(output_data.astype(np.float64)) == 1, "Contrast stretching failed (max)"

        if method == "adaptive_eq":
            assert np.all(output_data >= 0) and np.all(
                output_data <= 1), "Adaptive equalization failed normalization check"

def test_cleanup():
    if os.path.exists(tilt_stack_1_path):
        os.remove(tilt_stack_1_path)
    if os.path.exists(tilt_stack_2_path):
        os.remove(tilt_stack_2_path)
    if os.path.exists(str(Path(__file__).parent / "test_data" / "test_split_odd.mrc")):
        os.remove(str(Path(__file__).parent / "test_data" / "test_split_odd.mrc"))
    if os.path.exists(str(Path(__file__).parent / "test_data" / "test_split_even.mrc")):
        os.remove(str(Path(__file__).parent / "test_data" / "test_split_even.mrc"))
    if os.path.exists(str(Path(__file__).parent / "test_data" / "test_minus1_split.mrc")):
        os.remove(str(Path(__file__).parent / "test_data" / "test_minus1_split.mrc"))
    if os.path.exists(str(Path(__file__).parent / "test_data" / "test_merge")):
        shutil.rmtree(str(Path(__file__).parent / "test_data" / "test_merge"))
    if os.path.exists(flipped_path):
        os.remove(flipped_path)
    if os.path.exists(str(Path(__file__).parent / "test_data" / "expected_output.mrc")):
        os.remove(str(Path(__file__).parent / "test_data" / "expected_output.mrc"))

