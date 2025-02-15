import numpy as np
import pandas as pd

from cryocat.ioutils import *
import tempfile
import pytest
import os
import re
from io import StringIO
from pathlib import Path
import json

# function to create a temporary file with a specific encoding
def create_temp_file_with_encoding(content, encoding):
    """Creates a temporary file with the specified encoding."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding=encoding)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name
#not working yet: iso88591 and windows1252 are very similar?
@pytest.mark.parametrize("content, encoding, expected_encoding", [
    ("hello, this is a test.", "utf-8", "utf-8"),
    ("hello, this is a test with é.", "iso-8859-1", "iso-8859-1"),  # Added accented character for testing
    ("hello, this is a test with € symbol.", "windows-1252", "windows-1252"),  # Euro symbol for testing
])
def test_get_file_encoding_valid(content, encoding, expected_encoding):
    """Test the get_file_encoding function for various encodings."""
    # Create the temporary file with the specified encoding
    temp_file_path = create_temp_file_with_encoding(content, encoding)

    # Verify that the file's content matches what was written
    with open(temp_file_path, 'r', encoding=encoding) as f:
        file_content = f.read()
        assert file_content == content  # Make sure the content is as expected

    # Check if get_file_encoding detects the encoding correctly
    detected_encoding = get_file_encoding(temp_file_path)
    print(f"Detected encoding: {detected_encoding}")

    # Validate the detected encoding matches the expected encoding
    assert detected_encoding == expected_encoding

    # Clean up the temporary file after the test
    os.remove(temp_file_path)

#catch ValueError for not existing dir?
def test_get_files_prefix_suffix(tmp_path):
    temp_dir = tmp_path
    file_names = [
        "test_file1.txt",
        "test_file2.txt",
        "example_file1.txt",
        "test_file1.csv",
        "sample_file2.txt",
        "test_data.doc",
    ]
    for file_name in file_names:
        (temp_dir / file_name).touch()
    #case 1: Filter by prefix "test" and suffix ".txt"
    result = get_files_prefix_suffix(temp_dir, prefix="test", suffix=".txt")
    assert result == ["test_file1.txt", "test_file2.txt"]
    #case 2: Filter by prefix "example"
    result = get_files_prefix_suffix(temp_dir, prefix="example")
    assert result == ["example_file1.txt"]
    #case 3: Filter by suffix ".csv"
    result = get_files_prefix_suffix(temp_dir, suffix=".csv")
    assert result == ["test_file1.csv"]
    #case 4: No filtering (empty prefix and suffix)
    result = get_files_prefix_suffix(temp_dir)
    assert result == sorted(file_names)
    #case 5: Prefix or suffix that matches no files
    result = get_files_prefix_suffix(temp_dir, prefix="nonexistent")
    assert result == []
    result = get_files_prefix_suffix(temp_dir, suffix=".nonexistent")
    assert result == []
    #case 6:
    result = get_files_prefix_suffix(temp_dir, prefix="test")
    assert result == sorted(["test_file1.txt", "test_file2.txt", "test_data.doc", "test_file1.csv"])

#catch TypeError exception, added check for booleans?
def test_is_float():
    # valid float inputs
    assert is_float(3.14) == True
    assert is_float(0.0) == True
    assert is_float(-5.67) == True
    assert is_float("3.14") == True
    # not valid inputs
    assert is_float("hello") == False
    assert is_float(None) == False  # None value
    assert is_float("123abc") == False
    assert is_float([]) == False  #raise exception
    assert is_float({}) == False
    assert is_float(True) == False
    assert is_float(False) == False


def test_get_filename_from_path():
    #filename with extension
    input_path = "/home/user/documents/file.txt"
    expected = "file.txt"
    assert get_filename_from_path(input_path) == expected
    #filename without extension
    input_path = "/home/user/documents/file.txt"
    expected = "file"
    assert get_filename_from_path(input_path, with_extension=False) == expected
    #filename with no extension
    input_path = "file"
    expected = "file"
    assert get_filename_from_path(input_path) == expected
    #multiple directories
    input_path = "/home/user/documents/folder/subfolder/file.txt"
    expected = "file.txt"
    assert get_filename_from_path(input_path) == expected
    #multiple directories without extension
    input_path = "/home/user/documents/folder/subfolder/file.txt"
    expected = "file"
    assert get_filename_from_path(input_path, with_extension=False) == expected
    #with a file that has no extension
    input_path = "/home/user/documents/folder/subfolder/no_extension"
    expected = "no_extension"
    assert get_filename_from_path(input_path) == expected
    #containing a dot but no extension
    input_path = "/home/user/documents/file.withoutextension"
    expected = "file.withoutextension"
    assert get_filename_from_path(input_path) == expected
    #empty path
    input_path = ""
    expected = ""
    assert get_filename_from_path(input_path) == expected

    #leading/trailing spaces
    input_path = "  /home/user/documents/file.txt  "
    expected = "file.txt"
    assert get_filename_from_path(input_path.strip()) == expected

    #special characters in the filename
    input_path = "/home/user/documents/special@file#name!.txt"
    expected = "special@file#name!.txt"
    assert get_filename_from_path(input_path) == expected
    #unicode characters
    input_path = "/home/user/documents/file_éxample.txt"
    expected = "file_éxample.txt"
    assert get_filename_from_path(input_path) == expected
    #long file path
    input_path = "/home/user/documents/" + "a" * 255 + ".txt"
    expected = "a" * 255 + ".txt"
    assert get_filename_from_path(input_path) == expected

    #multiple dots but no extension
    input_path = "/home/user/documents/file.with.many.dots"
    expected = "file.with.many.dots"
    assert get_filename_from_path(input_path) == expected

    #TO ADD EXCEPTION, to check this edge case
    #not a string
    """
    input_path = None
    expected = None
    assert get_filename_from_path(input_path) == expected
    """

    #edge case
    input_path = "/home/user/documents/.hiddenfile.txt"
    expected = ".hiddenfile.txt"
    assert get_filename_from_path(input_path) == expected
    #containing only dots
    input_path = "/home/user/documents/...."
    expected = "...."
    assert get_filename_from_path(input_path) == expected

    #spaces in filename
    input_path = "/home/user/documents/file with spaces.txt"
    expected = "file with spaces.txt"
    assert get_filename_from_path(input_path) == expected


def test_get_number_of_lines_with_character():
    # create a temporary file for testing
    test_filename = "test_file.txt"

    # write test data to the file
    with open(test_filename, "w") as file:
        file.write("# this is a comment line\n")  # starts with '#'
        file.write("this is a normal line\n")  # does not start with '#'
        file.write("# another comment line\n")  # starts with '#'
        file.write("\n")  # empty line
        file.write("# yet another comment\n")  # starts with '#'
        file.write("last normal line\n")  # does not start with '#'
        file.write("some text with # in middle\n")  # '#' in middle, not at start

        #Spaces before the character
        file.write("  #\n")  # '#' after multiple spaces

        file.write("#\n")  # line with only '#'



    # test lines starting with the '#' character
    character = "#"
    expected = 4  # six lines start directly with '#'
    assert get_number_of_lines_with_character(test_filename, character) == expected


    # clean up temporary file
    os.remove(test_filename)

def test_fileformat_replace_pattern():
    #padding needed
    assert fileformat_replace_pattern("some_text_$AAA_rest", 79, "A") == "some_text_079_rest"
    #no padding needed
    assert fileformat_replace_pattern("file_/$AAA/$B.txt", 123, "A") == "file_/123/$B.txt"
    #number too big for pattern
    with pytest.raises(ValueError, match=re.escape("Number '12345' has more digits than string '$A'.")):
        fileformat_replace_pattern("file_/$A/$B.txt", 12345, "A")
    #pattern not present -raise exception
    with pytest.raises(ValueError, match=re.escape("The format file_/$A/$B.txt does not contain any sequence of \\$ followed by C.")):
        fileformat_replace_pattern("file_/$A/$B.txt", 123, "C")
    #no error if pattern is absent and raise_error false
    assert fileformat_replace_pattern("file_/$A/$B.txt", 123, "C", raise_error=False) == "file_/$A/$B.txt"
    #other multiple cases
    assert fileformat_replace_pattern("file_$AA_$BB_end", 7, "A") == "file_07_$BB_end"
    assert fileformat_replace_pattern("file_$AA_$BB_end", 5, "B") == "file_$AA_05_end"
    #complex input and multiple patterns
    assert fileformat_replace_pattern("path_$AAA/$BB/$CC_$DD.txt", 42, "A") == "path_042/$BB/$CC_$DD.txt"
    assert fileformat_replace_pattern("path_$AAA/$BB/$CC_$DD.txt", 3, "B") == "path_$AAA/03/$CC_$DD.txt"
    #single letter
    assert fileformat_replace_pattern("example_$A.txt", 2, "A") == "example_2.txt"
    #more than 1 pattern, what happens
    assert fileformat_replace_pattern("example_$AA/$A/$B.txt", 2, "A") == "example_02/2/$B.txt"


#Doesn't handle reading floats and booleans
#We are not interested into <Param> which contain just configurations
#Raise exception if node level != 1,2 ?
def test_get_data_from_warp_xml():
    current_dir = Path(__file__).parent
    xml_file_path = str(current_dir / "test_data" / "TS_017" / "017.xml")

    #Level 1, Angles, integers
    angles = np.array([-52, -50, -48, -46, -44, -42, -40, -38, -36, -34, -32, -30, -28,
           -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2,
           0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
           26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
           52, 54, 56, 58, 60, 62, 64, 66])
    assert np.array_equal(get_data_from_warp_xml(xml_file_path, "Angles", 1), angles)
    #Level 1, Dose, floats
    dose = np.array([134.3059 , 132.0675 , 127.5906 , 125.3522 , 118.6369 , 116.3985 ,
       109.6832 , 107.4447 , 100.7294 ,  98.49098,  91.77568,  89.53725,
        82.82195,  80.58351,  73.86821,  71.62978,  64.91447,  62.67604,
        55.96075,  53.72232,  47.00703,  44.7686 ,  38.05331,  35.81488,
        29.09959,  26.86116,  20.14587,  17.90744,  11.19215,   8.95372,
         2.23843,   4.47686,   6.71529,  13.43058,  15.66901,  22.3843 ,
        24.62273,  31.33802,  33.57645,  40.29174,  42.53017,  49.24546,
        51.48389,  58.19918,  60.43761,  67.15291,  69.39134,  76.10664,
        78.34508,  85.06038,  87.29881,  94.01411,  96.25255, 102.9678 ,
       105.2063 , 111.9216 , 114.16   , 120.8753 , 123.1138 , 129.8291 ])
    assert np.array_equal(get_data_from_warp_xml(xml_file_path, "Dose", 1), dose)


    #level 2, valid copy-paste node values
    nodedatatest = """
            <Node X="0" Y="0" Z="0" Value="-1.463246" />
        		<Node X="1" Y="0" Z="0" Value="5.320158" />
        		<Node X="2" Y="0" Z="0" Value="1.452159" />
        		<Node X="0" Y="1" Z="0" Value="-2.376527" />
        		<Node X="1" Y="1" Z="0" Value="-1.308048" />
        		<Node X="2" Y="1" Z="0" Value="1.455077" />
        		<Node X="0" Y="2" Z="0" Value="0.3901345" />
        		<Node X="1" Y="2" Z="0" Value="-0.5754877" />
        		<Node X="2" Y="2" Z="0" Value="-0.4075638" />
        		<Node X="0" Y="0" Z="1" Value="-0.7106043" />
        		<Node X="1" Y="0" Z="1" Value="5.864784" />
        		<Node X="2" Y="0" Z="1" Value="3.30233" />
        		<Node X="0" Y="1" Z="1" Value="-1.52878" />
        		<Node X="1" Y="1" Z="1" Value="-2.483566" />
        		<Node X="2" Y="1" Z="1" Value="-4.079008" />
        		<Node X="0" Y="2" Z="1" Value="0.2063846" />
        		<Node X="1" Y="2" Z="1" Value="-0.6359377" />
        		<Node X="2" Y="2" Z="1" Value="-0.4039657" />
        		<Node X="0" Y="0" Z="2" Value="-0.4095204" />
        		<Node X="1" Y="0" Z="2" Value="1.397589" />
        		<Node X="2" Y="0" Z="2" Value="0.2903593" />
        		<Node X="0" Y="1" Z="2" Value="0.185078" />
        		<Node X="1" Y="1" Z="2" Value="-1.683118" />
        		<Node X="2" Y="1" Z="2" Value="-2.564194" />
        		<Node X="0" Y="2" Z="2" Value="0.05996355" />
        		<Node X="1" Y="2" Z="2" Value="-0.08808023" />
        		<Node X="2" Y="2" Z="2" Value="0.03305629" />
        		<Node X="0" Y="0" Z="3" Value="-0.5266708" />
        		<Node X="1" Y="0" Z="3" Value="0.4416825" />
        		<Node X="2" Y="0" Z="3" Value="1.395934" />
        		<Node X="0" Y="1" Z="3" Value="-1.08517" />
        		<Node X="1" Y="1" Z="3" Value="-1.230583" />
        		<Node X="2" Y="1" Z="3" Value="-4.273524" />
        		<Node X="0" Y="2" Z="3" Value="0.1407896" />
        		<Node X="1" Y="2" Z="3" Value="0.2722078" />
        		<Node X="2" Y="2" Z="3" Value="0.224728" />
        		<Node X="0" Y="0" Z="4" Value="-0.9587417" />
        		<Node X="1" Y="0" Z="4" Value="-0.2396259" />
        		<Node X="2" Y="0" Z="4" Value="1.009668" />
        		<Node X="0" Y="1" Z="4" Value="-1.290259" />
        		<Node X="1" Y="1" Z="4" Value="-1.248055" />
        		<Node X="2" Y="1" Z="4" Value="-4.197393" />
        		<Node X="0" Y="2" Z="4" Value="0.2185136" />
        		<Node X="1" Y="2" Z="4" Value="-0.2632738" />
        		<Node X="2" Y="2" Z="4" Value="0.06670134" />
        		<Node X="0" Y="0" Z="5" Value="-0.3779781" />
        		<Node X="1" Y="0" Z="5" Value="-1.635345" />
        		<Node X="2" Y="0" Z="5" Value="0.3110349" />
        		<Node X="0" Y="1" Z="5" Value="-0.7143888" />
        		<Node X="1" Y="1" Z="5" Value="-1.022123" />
        		<Node X="2" Y="1" Z="5" Value="-4.777106" />
        		<Node X="0" Y="2" Z="5" Value="0.08590294" />
        		<Node X="1" Y="2" Z="5" Value="0.2265385" />
        		<Node X="2" Y="2" Z="5" Value="0.1821524" />
        		<Node X="0" Y="0" Z="6" Value="-1.530601" />
        		<Node X="1" Y="0" Z="6" Value="-0.5546232" />
        		<Node X="2" Y="0" Z="6" Value="1.850821" />
        		<Node X="0" Y="1" Z="6" Value="-1.653455" />
        		<Node X="1" Y="1" Z="6" Value="-0.7825686" />
        		<Node X="2" Y="1" Z="6" Value="-5.108892" />
        		<Node X="0" Y="2" Z="6" Value="0.3053687" />
        		<Node X="1" Y="2" Z="6" Value="-0.1588862" />
        		<Node X="2" Y="2" Z="6" Value="-0.3030559" />
        		<Node X="0" Y="0" Z="7" Value="-0.3694279" />
        		<Node X="1" Y="0" Z="7" Value="-3.079624" />
        		<Node X="2" Y="0" Z="7" Value="-1.713288" />
        		<Node X="0" Y="1" Z="7" Value="-1.834426" />
        		<Node X="1" Y="1" Z="7" Value="-1.526223" />
        		<Node X="2" Y="1" Z="7" Value="-2.180725" />
        		<Node X="0" Y="2" Z="7" Value="0.1993298" />
        		<Node X="1" Y="2" Z="7" Value="0.3071583" />
        		<Node X="2" Y="2" Z="7" Value="0.3434351" />
        		<Node X="0" Y="0" Z="8" Value="0.7162532" />
        		<Node X="1" Y="0" Z="8" Value="-2.697225" />
        		<Node X="2" Y="0" Z="8" Value="-1.586078" />
        		<Node X="0" Y="1" Z="8" Value="0.321272" />
        		<Node X="1" Y="1" Z="8" Value="-1.675499" />
        		<Node X="2" Y="1" Z="8" Value="-1.831161" />
        		<Node X="0" Y="2" Z="8" Value="-0.1197902" />
        		<Node X="1" Y="2" Z="8" Value="0.3940544" />
        		<Node X="2" Y="2" Z="8" Value="0.3820015" />
        		<Node X="0" Y="0" Z="9" Value="-0.4948481" />
        		<Node X="1" Y="0" Z="9" Value="0.01214749" />
        		<Node X="2" Y="0" Z="9" Value="-1.277677" />
        		<Node X="0" Y="1" Z="9" Value="-2.367667" />
        		<Node X="1" Y="1" Z="9" Value="-2.466241" />
        		<Node X="2" Y="1" Z="9" Value="-2.800977" />
        		<Node X="0" Y="2" Z="9" Value="0.2596546" />
        		<Node X="1" Y="2" Z="9" Value="0.3332871" />
        		<Node X="2" Y="2" Z="9" Value="0.1657559" />
        		<Node X="0" Y="0" Z="10" Value="0.2982151" />
        		<Node X="1" Y="0" Z="10" Value="-1.50164" />
        		<Node X="2" Y="0" Z="10" Value="-1.518515" />
        		<Node X="0" Y="1" Z="10" Value="-0.6211068" />
        		<Node X="1" Y="1" Z="10" Value="-2.937962" />
        		<Node X="2" Y="1" Z="10" Value="-4.469218" />
        		<Node X="0" Y="2" Z="10" Value="0.01148285" />
        		<Node X="1" Y="2" Z="10" Value="0.7165185" />
        		<Node X="2" Y="2" Z="10" Value="0.5784123" />
        		<Node X="0" Y="0" Z="11" Value="-0.5220225" />
        		<Node X="1" Y="0" Z="11" Value="-3.54739" />
        		<Node X="2" Y="0" Z="11" Value="-0.470219" />
        		<Node X="0" Y="1" Z="11" Value="-1.295103" />
        		<Node X="1" Y="1" Z="11" Value="-1.014738" />
        		<Node X="2" Y="1" Z="11" Value="-1.174498" />
        		<Node X="0" Y="2" Z="11" Value="0.1662981" />
        		<Node X="1" Y="2" Z="11" Value="0.4758431" />
        		<Node X="2" Y="2" Z="11" Value="-0.009146304" />
        		<Node X="0" Y="0" Z="12" Value="-1.012112" />
        		<Node X="1" Y="0" Z="12" Value="-3.174333" />
        		<Node X="2" Y="0" Z="12" Value="0.8814076" />
        		<Node X="0" Y="1" Z="12" Value="-1.22106" />
        		<Node X="1" Y="1" Z="12" Value="-1.344784" />
        		<Node X="2" Y="1" Z="12" Value="-3.342522" />
        		<Node X="0" Y="2" Z="12" Value="0.1792351" />
        		<Node X="1" Y="2" Z="12" Value="0.2866584" />
        		<Node X="2" Y="2" Z="12" Value="-0.07647342" />
        		<Node X="0" Y="0" Z="13" Value="-0.4943229" />
        		<Node X="1" Y="0" Z="13" Value="-1.95817" />
        		<Node X="2" Y="0" Z="13" Value="-0.6274569" />
        		<Node X="0" Y="1" Z="13" Value="-1.629542" />
        		<Node X="1" Y="1" Z="13" Value="-1.810308" />
        		<Node X="2" Y="1" Z="13" Value="-2.223524" />
        		<Node X="0" Y="2" Z="13" Value="0.142798" />
        		<Node X="1" Y="2" Z="13" Value="0.5942734" />
        		<Node X="2" Y="2" Z="13" Value="0.3156306" />
        		<Node X="0" Y="0" Z="14" Value="0.1230368" />
        		<Node X="1" Y="0" Z="14" Value="-1.255964" />
        		<Node X="2" Y="0" Z="14" Value="-2.560725" />
        		<Node X="0" Y="1" Z="14" Value="-0.2168628" />
        		<Node X="1" Y="1" Z="14" Value="-2.68361" />
        		<Node X="2" Y="1" Z="14" Value="-2.556492" />
        		<Node X="0" Y="2" Z="14" Value="-0.01003722" />
        		<Node X="1" Y="2" Z="14" Value="0.2361742" />
        		<Node X="2" Y="2" Z="14" Value="0.4976555" />
        		<Node X="0" Y="0" Z="15" Value="-1.611152" />
        		<Node X="1" Y="0" Z="15" Value="-2.151034" />
        		<Node X="2" Y="0" Z="15" Value="-2.108429" />
        		<Node X="0" Y="1" Z="15" Value="-1.768244" />
        		<Node X="1" Y="1" Z="15" Value="-1.228219" />
        		<Node X="2" Y="1" Z="15" Value="-3.732646" />
        		<Node X="0" Y="2" Z="15" Value="0.2576976" />
        		<Node X="1" Y="2" Z="15" Value="0.3873455" />
        		<Node X="2" Y="2" Z="15" Value="0.5294612" />
        		<Node X="0" Y="0" Z="16" Value="-0.6109841" />
        		<Node X="1" Y="0" Z="16" Value="-0.681752" />
        		<Node X="2" Y="0" Z="16" Value="-1.912903" />
        		<Node X="0" Y="1" Z="16" Value="-1.016887" />
        		<Node X="1" Y="1" Z="16" Value="-1.598863" />
        		<Node X="2" Y="1" Z="16" Value="-3.753352" />
        		<Node X="0" Y="2" Z="16" Value="0.1342393" />
        		<Node X="1" Y="2" Z="16" Value="0.2953961" />
        		<Node X="2" Y="2" Z="16" Value="0.2839231" />
        		<Node X="0" Y="0" Z="17" Value="-0.315821" />
        		<Node X="1" Y="0" Z="17" Value="-0.9707435" />
        		<Node X="2" Y="0" Z="17" Value="-0.00040579" />
        		<Node X="0" Y="1" Z="17" Value="-1.032178" />
        		<Node X="1" Y="1" Z="17" Value="-0.2488512" />
        		<Node X="2" Y="1" Z="17" Value="-4.106051" />
        		<Node X="0" Y="2" Z="17" Value="0.1406045" />
        		<Node X="1" Y="2" Z="17" Value="0.1552727" />
        		<Node X="2" Y="2" Z="17" Value="-0.01827077" />
        		<Node X="0" Y="0" Z="18" Value="0.3701774" />
        		<Node X="1" Y="0" Z="18" Value="-1.723352" />
        		<Node X="2" Y="0" Z="18" Value="-1.372155" />
        		<Node X="0" Y="1" Z="18" Value="-0.8536774" />
        		<Node X="1" Y="1" Z="18" Value="-0.56396" />
        		<Node X="2" Y="1" Z="18" Value="-2.582149" />
        		<Node X="0" Y="2" Z="18" Value="0.1191429" />
        		<Node X="1" Y="2" Z="18" Value="0.2179693" />
        		<Node X="2" Y="2" Z="18" Value="-0.07021593" />
        		<Node X="0" Y="0" Z="19" Value="-0.1060382" />
        		<Node X="1" Y="0" Z="19" Value="-1.811532" />
        		<Node X="2" Y="0" Z="19" Value="-1.88311" />
        		<Node X="0" Y="1" Z="19" Value="-0.4577783" />
        		<Node X="1" Y="1" Z="19" Value="-0.09073634" />
        		<Node X="2" Y="1" Z="19" Value="-1.782525" />
        		<Node X="0" Y="2" Z="19" Value="0.06478995" />
        		<Node X="1" Y="2" Z="19" Value="0.2026641" />
        		<Node X="2" Y="2" Z="19" Value="0.09382563" />
        		<Node X="0" Y="0" Z="20" Value="-0.08359869" />
        		<Node X="1" Y="0" Z="20" Value="-0.7033283" />
        		<Node X="2" Y="0" Z="20" Value="-1.705231" />
        		<Node X="0" Y="1" Z="20" Value="-0.3581509" />
        		<Node X="1" Y="1" Z="20" Value="-1.204437" />
        		<Node X="2" Y="1" Z="20" Value="-1.552996" />
        		<Node X="0" Y="2" Z="20" Value="0.03866892" />
        		<Node X="1" Y="2" Z="20" Value="0.06212883" />
        		<Node X="2" Y="2" Z="20" Value="0.1897697" />
        		<Node X="0" Y="0" Z="21" Value="-1.195053" />
        		<Node X="1" Y="0" Z="21" Value="-1.273353" />
        		<Node X="2" Y="0" Z="21" Value="0.1509104" />
        		<Node X="0" Y="1" Z="21" Value="-0.6179267" />
        		<Node X="1" Y="1" Z="21" Value="-0.2443093" />
        		<Node X="2" Y="1" Z="21" Value="-2.65915" />
        		<Node X="0" Y="2" Z="21" Value="0.1130377" />
        		<Node X="1" Y="2" Z="21" Value="-0.0814635" />
        		<Node X="2" Y="2" Z="21" Value="-0.1326908" />
        		<Node X="0" Y="0" Z="22" Value="0.008384475" />
        		<Node X="1" Y="0" Z="22" Value="-1.716682" />
        		<Node X="2" Y="0" Z="22" Value="-1.825313" />
        		<Node X="0" Y="1" Z="22" Value="0.6198087" />
        		<Node X="1" Y="1" Z="22" Value="0.210767" />
        		<Node X="2" Y="1" Z="22" Value="-2.328422" />
        		<Node X="0" Y="2" Z="22" Value="-0.05069068" />
        		<Node X="1" Y="2" Z="22" Value="-0.09321602" />
        		<Node X="2" Y="2" Z="22" Value="-0.009880386" />
        		<Node X="0" Y="0" Z="23" Value="-0.4430967" />
        		<Node X="1" Y="0" Z="23" Value="-1.305517" />
        		<Node X="2" Y="0" Z="23" Value="-1.719509" />
        		<Node X="0" Y="1" Z="23" Value="0.1801253" />
        		<Node X="1" Y="1" Z="23" Value="-0.4872607" />
        		<Node X="2" Y="1" Z="23" Value="-1.736029" />
        		<Node X="0" Y="2" Z="23" Value="-0.01621913" />
        		<Node X="1" Y="2" Z="23" Value="0.04965311" />
        		<Node X="2" Y="2" Z="23" Value="0.1059583" />
        		<Node X="0" Y="0" Z="24" Value="-0.15272" />
        		<Node X="1" Y="0" Z="24" Value="-1.140445" />
        		<Node X="2" Y="0" Z="24" Value="-1.187778" />
        		<Node X="0" Y="1" Z="24" Value="-0.8995698" />
        		<Node X="1" Y="1" Z="24" Value="0.1290679" />
        		<Node X="2" Y="1" Z="24" Value="-2.039777" />
        		<Node X="0" Y="2" Z="24" Value="0.1745232" />
        		<Node X="1" Y="2" Z="24" Value="-0.2871963" />
        		<Node X="2" Y="2" Z="24" Value="-0.1064552" />
        		<Node X="0" Y="0" Z="25" Value="1.473913" />
        		<Node X="1" Y="0" Z="25" Value="-0.6776252" />
        		<Node X="2" Y="0" Z="25" Value="-1.987762" />
        		<Node X="0" Y="1" Z="25" Value="0.2574184" />
        		<Node X="1" Y="1" Z="25" Value="0.1255547" />
        		<Node X="2" Y="1" Z="25" Value="0.2881294" />
        		<Node X="0" Y="2" Z="25" Value="-0.04728567" />
        		<Node X="1" Y="2" Z="25" Value="-0.05159292" />
        		<Node X="2" Y="2" Z="25" Value="-0.11628" />
        		<Node X="0" Y="0" Z="26" Value="0.01599271" />
        		<Node X="1" Y="0" Z="26" Value="-0.463193" />
        		<Node X="2" Y="0" Z="26" Value="-1.147276" />
        		<Node X="0" Y="1" Z="26" Value="2.109198" />
        		<Node X="1" Y="1" Z="26" Value="0.1329704" />
        		<Node X="2" Y="1" Z="26" Value="-1.443648" />
        		<Node X="0" Y="2" Z="26" Value="-0.2384964" />
        		<Node X="1" Y="2" Z="26" Value="-0.1099772" />
        		<Node X="2" Y="2" Z="26" Value="0.3359437" />
        		<Node X="0" Y="0" Z="27" Value="-0.09431294" />
        		<Node X="1" Y="0" Z="27" Value="-0.7216987" />
        		<Node X="2" Y="0" Z="27" Value="-0.4680083" />
        		<Node X="0" Y="1" Z="27" Value="0.6385919" />
        		<Node X="1" Y="1" Z="27" Value="0.2349623" />
        		<Node X="2" Y="1" Z="27" Value="-0.6990215" />
        		<Node X="0" Y="2" Z="27" Value="-0.07671299" />
        		<Node X="1" Y="2" Z="27" Value="-0.1488184" />
        		<Node X="2" Y="2" Z="27" Value="0.1199325" />
        		<Node X="0" Y="0" Z="28" Value="0.06634519" />
        		<Node X="1" Y="0" Z="28" Value="-0.2920912" />
        		<Node X="2" Y="0" Z="28" Value="-0.5589179" />
        		<Node X="0" Y="1" Z="28" Value="1.428783" />
        		<Node X="1" Y="1" Z="28" Value="0.3591155" />
        		<Node X="2" Y="1" Z="28" Value="0.9722123" />
        		<Node X="0" Y="2" Z="28" Value="-0.2322642" />
        		<Node X="1" Y="2" Z="28" Value="0.1640536" />
        		<Node X="2" Y="2" Z="28" Value="0.3133739" />
        		<Node X="0" Y="0" Z="29" Value="0.7822325" />
        		<Node X="1" Y="0" Z="29" Value="0.3300999" />
        		<Node X="2" Y="0" Z="29" Value="-1.627798" />
        		<Node X="0" Y="1" Z="29" Value="-0.1763324" />
        		<Node X="1" Y="1" Z="29" Value="0.6838462" />
        		<Node X="2" Y="1" Z="29" Value="1.724693" />
        		<Node X="0" Y="2" Z="29" Value="-0.00318099" />
        		<Node X="1" Y="2" Z="29" Value="-0.02396142" />
        		<Node X="2" Y="2" Z="29" Value="0.1055321" />
        		<Node X="0" Y="0" Z="30" Value="0.1433068" />
        		<Node X="1" Y="0" Z="30" Value="1.349256" />
        		<Node X="2" Y="0" Z="30" Value="2.26653" />
        		<Node X="0" Y="1" Z="30" Value="1.422405" />
        		<Node X="1" Y="1" Z="30" Value="2.410702" />
        		<Node X="2" Y="1" Z="30" Value="3.195401" />
        		<Node X="0" Y="2" Z="30" Value="-0.2058323" />
        		<Node X="1" Y="2" Z="30" Value="-0.4314051" />
        		<Node X="2" Y="2" Z="30" Value="-0.1900425" />
        		<Node X="0" Y="0" Z="31" Value="-1.179764" />
        		<Node X="1" Y="0" Z="31" Value="0.883917" />
        		<Node X="2" Y="0" Z="31" Value="1.174489" />
        		<Node X="0" Y="1" Z="31" Value="0.09098724" />
        		<Node X="1" Y="1" Z="31" Value="2.660588" />
        		<Node X="2" Y="1" Z="31" Value="2.385371" />
        		<Node X="0" Y="2" Z="31" Value="0.03932115" />
        		<Node X="1" Y="2" Z="31" Value="-0.4811596" />
        		<Node X="2" Y="2" Z="31" Value="-0.1107998" />
        		<Node X="0" Y="0" Z="32" Value="-0.6113384" />
        		<Node X="1" Y="0" Z="32" Value="0.6766121" />
        		<Node X="2" Y="0" Z="32" Value="0.9443513" />
        		<Node X="0" Y="1" Z="32" Value="0.8643097" />
        		<Node X="1" Y="1" Z="32" Value="1.983773" />
        		<Node X="2" Y="1" Z="32" Value="2.509074" />
        		<Node X="0" Y="2" Z="32" Value="-0.1268053" />
        		<Node X="1" Y="2" Z="32" Value="-0.2978492" />
        		<Node X="2" Y="2" Z="32" Value="0.2010989" />
        		<Node X="0" Y="0" Z="33" Value="0.2941595" />
        		<Node X="1" Y="0" Z="33" Value="-0.003231251" />
        		<Node X="2" Y="0" Z="33" Value="0.4881998" />
        		<Node X="0" Y="1" Z="33" Value="1.208933" />
        		<Node X="1" Y="1" Z="33" Value="1.942395" />
        		<Node X="2" Y="1" Z="33" Value="1.338014" />
        		<Node X="0" Y="2" Z="33" Value="-0.1935601" />
        		<Node X="1" Y="2" Z="33" Value="-0.3215258" />
        		<Node X="2" Y="2" Z="33" Value="-0.05696579" />
        		<Node X="0" Y="0" Z="34" Value="-0.3472976" />
        		<Node X="1" Y="0" Z="34" Value="0.09433644" />
        		<Node X="2" Y="0" Z="34" Value="0.3397444" />
        		<Node X="0" Y="1" Z="34" Value="0.127166" />
        		<Node X="1" Y="1" Z="34" Value="1.557775" />
        		<Node X="2" Y="1" Z="34" Value="1.189507" />
        		<Node X="0" Y="2" Z="34" Value="-0.03112097" />
        		<Node X="1" Y="2" Z="34" Value="-0.1582837" />
        		<Node X="2" Y="2" Z="34" Value="0.216153" />
        		<Node X="0" Y="0" Z="35" Value="-0.03777345" />
        		<Node X="1" Y="0" Z="35" Value="-0.4355727" />
        		<Node X="2" Y="0" Z="35" Value="1.36326" />
        		<Node X="0" Y="1" Z="35" Value="0.3529037" />
        		<Node X="1" Y="1" Z="35" Value="1.915063" />
        		<Node X="2" Y="1" Z="35" Value="0.5705025" />
        		<Node X="0" Y="2" Z="35" Value="-0.05825124" />
        		<Node X="1" Y="2" Z="35" Value="-0.1032214" />
        		<Node X="2" Y="2" Z="35" Value="0.3005217" />
        		<Node X="0" Y="0" Z="36" Value="-0.8399256" />
        		<Node X="1" Y="0" Z="36" Value="0.3645122" />
        		<Node X="2" Y="0" Z="36" Value="0.2628777" />
        		<Node X="0" Y="1" Z="36" Value="-0.01787591" />
        		<Node X="1" Y="1" Z="36" Value="1.024429" />
        		<Node X="2" Y="1" Z="36" Value="1.444087" />
        		<Node X="0" Y="2" Z="36" Value="-0.01184748" />
        		<Node X="1" Y="2" Z="36" Value="0.02578417" />
        		<Node X="2" Y="2" Z="36" Value="0.06493462" />
        		<Node X="0" Y="0" Z="37" Value="0.3468409" />
        		<Node X="1" Y="0" Z="37" Value="-0.5145742" />
        		<Node X="2" Y="0" Z="37" Value="0.6361068" />
        		<Node X="0" Y="1" Z="37" Value="1.043736" />
        		<Node X="1" Y="1" Z="37" Value="0.5854191" />
        		<Node X="2" Y="1" Z="37" Value="1.211399" />
        		<Node X="0" Y="2" Z="37" Value="-0.1399718" />
        		<Node X="1" Y="2" Z="37" Value="-0.1381016" />
        		<Node X="2" Y="2" Z="37" Value="-0.1884141" />
        		<Node X="0" Y="0" Z="38" Value="-0.3449529" />
        		<Node X="1" Y="0" Z="38" Value="0.01692064" />
        		<Node X="2" Y="0" Z="38" Value="1.315026" />
        		<Node X="0" Y="1" Z="38" Value="-0.8521376" />
        		<Node X="1" Y="1" Z="38" Value="1.14969" />
        		<Node X="2" Y="1" Z="38" Value="1.065405" />
        		<Node X="0" Y="2" Z="38" Value="0.1287372" />
        		<Node X="1" Y="2" Z="38" Value="-0.2063174" />
        		<Node X="2" Y="2" Z="38" Value="-0.4680429" />
        		<Node X="0" Y="0" Z="39" Value="0.2715719" />
        		<Node X="1" Y="0" Z="39" Value="-0.8855376" />
        		<Node X="2" Y="0" Z="39" Value="3.277173" />
        		<Node X="0" Y="1" Z="39" Value="0.2681186" />
        		<Node X="1" Y="1" Z="39" Value="1.410705" />
        		<Node X="2" Y="1" Z="39" Value="0.6290808" />
        		<Node X="0" Y="2" Z="39" Value="-0.06141897" />
        		<Node X="1" Y="2" Z="39" Value="-0.09191106" />
        		<Node X="2" Y="2" Z="39" Value="-0.04201572" />
        		<Node X="0" Y="0" Z="40" Value="-0.1328561" />
        		<Node X="1" Y="0" Z="40" Value="-0.6741854" />
        		<Node X="2" Y="0" Z="40" Value="0.6705673" />
        		<Node X="0" Y="1" Z="40" Value="-0.525125" />
        		<Node X="1" Y="1" Z="40" Value="1.498924" />
        		<Node X="2" Y="1" Z="40" Value="0.7334902" />
        		<Node X="0" Y="2" Z="40" Value="0.06934692" />
        		<Node X="1" Y="2" Z="40" Value="-0.04914386" />
        		<Node X="2" Y="2" Z="40" Value="-0.4205783" />
        		<Node X="0" Y="0" Z="41" Value="-0.2773003" />
        		<Node X="1" Y="0" Z="41" Value="-0.9362093" />
        		<Node X="2" Y="0" Z="41" Value="-0.2420284" />
        		<Node X="0" Y="1" Z="41" Value="0.0848609" />
        		<Node X="1" Y="1" Z="41" Value="0.489738" />
        		<Node X="2" Y="1" Z="41" Value="1.853848" />
        		<Node X="0" Y="2" Z="41" Value="0.01720244" />
        		<Node X="1" Y="2" Z="41" Value="-0.07237782" />
        		<Node X="2" Y="2" Z="41" Value="-0.4250088" />
        		<Node X="0" Y="0" Z="42" Value="0.5253142" />
        		<Node X="1" Y="0" Z="42" Value="-1.027279" />
        		<Node X="2" Y="0" Z="42" Value="0.1768653" />
        		<Node X="0" Y="1" Z="42" Value="1.107163" />
        		<Node X="1" Y="1" Z="42" Value="1.797172" />
        		<Node X="2" Y="1" Z="42" Value="1.85055" />
        		<Node X="0" Y="2" Z="42" Value="-0.1254745" />
        		<Node X="1" Y="2" Z="42" Value="-0.2504082" />
        		<Node X="2" Y="2" Z="42" Value="-0.2743937" />
        		<Node X="0" Y="0" Z="43" Value="0.1132936" />
        		<Node X="1" Y="0" Z="43" Value="-1.633733" />
        		<Node X="2" Y="0" Z="43" Value="0.5591072" />
        		<Node X="0" Y="1" Z="43" Value="0.7104746" />
        		<Node X="1" Y="1" Z="43" Value="1.830815" />
        		<Node X="2" Y="1" Z="43" Value="1.436787" />
        		<Node X="0" Y="2" Z="43" Value="-0.1145731" />
        		<Node X="1" Y="2" Z="43" Value="0.1250063" />
        		<Node X="2" Y="2" Z="43" Value="-0.157924" />
        		<Node X="0" Y="0" Z="44" Value="-0.349147" />
        		<Node X="1" Y="0" Z="44" Value="0.3923524" />
        		<Node X="2" Y="0" Z="44" Value="-0.2887461" />
        		<Node X="0" Y="1" Z="44" Value="0.1901588" />
        		<Node X="1" Y="1" Z="44" Value="1.04338" />
        		<Node X="2" Y="1" Z="44" Value="1.554292" />
        		<Node X="0" Y="2" Z="44" Value="0.01251187" />
        		<Node X="1" Y="2" Z="44" Value="-0.07875831" />
        		<Node X="2" Y="2" Z="44" Value="-0.3525737" />
        		<Node X="0" Y="0" Z="45" Value="0.2201274" />
        		<Node X="1" Y="0" Z="45" Value="-2.493914" />
        		<Node X="2" Y="0" Z="45" Value="0.7733946" />
        		<Node X="0" Y="1" Z="45" Value="-0.8519823" />
        		<Node X="1" Y="1" Z="45" Value="1.001191" />
        		<Node X="2" Y="1" Z="45" Value="2.241728" />
        		<Node X="0" Y="2" Z="45" Value="0.03651591" />
        		<Node X="1" Y="2" Z="45" Value="0.4484932" />
        		<Node X="2" Y="2" Z="45" Value="-0.06920049" />
        		<Node X="0" Y="0" Z="46" Value="-0.936515" />
        		<Node X="1" Y="0" Z="46" Value="-1.173076" />
        		<Node X="2" Y="0" Z="46" Value="1.238834" />
        		<Node X="0" Y="1" Z="46" Value="0.9181442" />
        		<Node X="1" Y="1" Z="46" Value="1.752505" />
        		<Node X="2" Y="1" Z="46" Value="0.5683882" />
        		<Node X="0" Y="2" Z="46" Value="-0.07035258" />
        		<Node X="1" Y="2" Z="46" Value="0.03659626" />
        		<Node X="2" Y="2" Z="46" Value="-0.02968399" />
        		<Node X="0" Y="0" Z="47" Value="0.1427311" />
        		<Node X="1" Y="0" Z="47" Value="-3.099199" />
        		<Node X="2" Y="0" Z="47" Value="-0.9049471" />
        		<Node X="0" Y="1" Z="47" Value="0.771832" />
        		<Node X="1" Y="1" Z="47" Value="1.965032" />
        		<Node X="2" Y="1" Z="47" Value="1.828468" />
        		<Node X="0" Y="2" Z="47" Value="-0.1411418" />
        		<Node X="1" Y="2" Z="47" Value="0.2229437" />
        		<Node X="2" Y="2" Z="47" Value="0.1171091" />
        		<Node X="0" Y="0" Z="48" Value="0.1909064" />
        		<Node X="1" Y="0" Z="48" Value="0.5024482" />
        		<Node X="2" Y="0" Z="48" Value="1.095386" />
        		<Node X="0" Y="1" Z="48" Value="0.3759831" />
        		<Node X="1" Y="1" Z="48" Value="1.345439" />
        		<Node X="2" Y="1" Z="48" Value="0.6294759" />
        		<Node X="0" Y="2" Z="48" Value="-0.02446398" />
        		<Node X="1" Y="2" Z="48" Value="-0.193266" />
        		<Node X="2" Y="2" Z="48" Value="-0.2045425" />
        		<Node X="0" Y="0" Z="49" Value="0.7162808" />
        		<Node X="1" Y="0" Z="49" Value="-2.214506" />
        		<Node X="2" Y="0" Z="49" Value="0.8649447" />
        		<Node X="0" Y="1" Z="49" Value="1.871275" />
        		<Node X="1" Y="1" Z="49" Value="2.490769" />
        		<Node X="2" Y="1" Z="49" Value="1.605241" />
        		<Node X="0" Y="2" Z="49" Value="-0.2157702" />
        		<Node X="1" Y="2" Z="49" Value="0.0398576" />
        		<Node X="2" Y="2" Z="49" Value="-0.30811" />
        		<Node X="0" Y="0" Z="50" Value="0.538143" />
        		<Node X="1" Y="0" Z="50" Value="-0.8824145" />
        		<Node X="2" Y="0" Z="50" Value="0.01733304" />
        		<Node X="0" Y="1" Z="50" Value="0.04796191" />
        		<Node X="1" Y="1" Z="50" Value="2.129442" />
        		<Node X="2" Y="1" Z="50" Value="2.682483" />
        		<Node X="0" Y="2" Z="50" Value="-0.02275831" />
        		<Node X="1" Y="2" Z="50" Value="-0.329757" />
        		<Node X="2" Y="2" Z="50" Value="-0.4441737" />
        		<Node X="0" Y="0" Z="51" Value="-0.07612614" />
        		<Node X="1" Y="0" Z="51" Value="-2.40694" />
        		<Node X="2" Y="0" Z="51" Value="0.03193626" />
        		<Node X="0" Y="1" Z="51" Value="-0.2932324" />
        		<Node X="1" Y="1" Z="51" Value="2.623614" />
        		<Node X="2" Y="1" Z="51" Value="1.631239" />
        		<Node X="0" Y="2" Z="51" Value="0.0364047" />
        		<Node X="1" Y="2" Z="51" Value="0.1970877" />
        		<Node X="2" Y="2" Z="51" Value="-0.1242779" />
        		<Node X="0" Y="0" Z="52" Value="-0.833669" />
        		<Node X="1" Y="0" Z="52" Value="-1.254515" />
        		<Node X="2" Y="0" Z="52" Value="-0.168173" />
        		<Node X="0" Y="1" Z="52" Value="-2.982485" />
        		<Node X="1" Y="1" Z="52" Value="3.099726" />
        		<Node X="2" Y="1" Z="52" Value="2.312814" />
        		<Node X="0" Y="2" Z="52" Value="0.2808829" />
        		<Node X="1" Y="2" Z="52" Value="0.2039993" />
        		<Node X="2" Y="2" Z="52" Value="-0.0898677" />
        		<Node X="0" Y="0" Z="53" Value="-0.3158733" />
        		<Node X="1" Y="0" Z="53" Value="-2.156559" />
        		<Node X="2" Y="0" Z="53" Value="2.417242" />
        		<Node X="0" Y="1" Z="53" Value="-1.047516" />
        		<Node X="1" Y="1" Z="53" Value="3.627437" />
        		<Node X="2" Y="1" Z="53" Value="1.944811" />
        		<Node X="0" Y="2" Z="53" Value="0.09728091" />
        		<Node X="1" Y="2" Z="53" Value="0.1532321" />
        		<Node X="2" Y="2" Z="53" Value="-0.6192255" />
        		<Node X="0" Y="0" Z="54" Value="-0.04899749" />
        		<Node X="1" Y="0" Z="54" Value="2.848426" />
        		<Node X="2" Y="0" Z="54" Value="0.2904806" />
        		<Node X="0" Y="1" Z="54" Value="-2.252651" />
        		<Node X="1" Y="1" Z="54" Value="2.962822" />
        		<Node X="2" Y="1" Z="54" Value="3.871899" />
        		<Node X="0" Y="2" Z="54" Value="0.1288705" />
        		<Node X="1" Y="2" Z="54" Value="-0.09419243" />
        		<Node X="2" Y="2" Z="54" Value="-0.1377179" />
        		<Node X="0" Y="0" Z="55" Value="0.8474329" />
        		<Node X="1" Y="0" Z="55" Value="1.214333" />
        		<Node X="2" Y="0" Z="55" Value="-0.52802" />
        		<Node X="0" Y="1" Z="55" Value="-0.4807324" />
        		<Node X="1" Y="1" Z="55" Value="4.788553" />
        		<Node X="2" Y="1" Z="55" Value="0.5269803" />
        		<Node X="0" Y="2" Z="55" Value="-0.134818" />
        		<Node X="1" Y="2" Z="55" Value="-0.2000996" />
        		<Node X="2" Y="2" Z="55" Value="0.07953535" />
        		<Node X="0" Y="0" Z="56" Value="0.1961866" />
        		<Node X="1" Y="0" Z="56" Value="0.630663" />
        		<Node X="2" Y="0" Z="56" Value="-1.200837" />
        		<Node X="0" Y="1" Z="56" Value="-1.504695" />
        		<Node X="1" Y="1" Z="56" Value="4.681966" />
        		<Node X="2" Y="1" Z="56" Value="3.99545" />
        		<Node X="0" Y="2" Z="56" Value="0.05390552" />
        		<Node X="1" Y="2" Z="56" Value="-0.2746132" />
        		<Node X="2" Y="2" Z="56" Value="0.09590138" />
        		<Node X="0" Y="0" Z="57" Value="-0.192025" />
        		<Node X="1" Y="0" Z="57" Value="2.120564" />
        		<Node X="2" Y="0" Z="57" Value="-0.5196728" />
        		<Node X="0" Y="1" Z="57" Value="-2.184315" />
        		<Node X="1" Y="1" Z="57" Value="3.931082" />
        		<Node X="2" Y="1" Z="57" Value="4.095085" />
        		<Node X="0" Y="2" Z="57" Value="0.1436728" />
        		<Node X="1" Y="2" Z="57" Value="-0.6218677" />
        		<Node X="2" Y="2" Z="57" Value="0.02701958" />
        		<Node X="0" Y="0" Z="58" Value="0.6309627" />
        		<Node X="1" Y="0" Z="58" Value="-0.869297" />
        		<Node X="2" Y="0" Z="58" Value="0.2767608" />
        		<Node X="0" Y="1" Z="58" Value="1.617473" />
        		<Node X="1" Y="1" Z="58" Value="5.58949" />
        		<Node X="2" Y="1" Z="58" Value="6.701704" />
        		<Node X="0" Y="2" Z="58" Value="-0.2136974" />
        		<Node X="1" Y="2" Z="58" Value="-0.151066" />
        		<Node X="2" Y="2" Z="58" Value="-0.08292479" />
        		<Node X="0" Y="0" Z="59" Value="-0.1033018" />
        		<Node X="1" Y="0" Z="59" Value="-0.8923529" />
        		<Node X="2" Y="0" Z="59" Value="1.542079" />
        		<Node X="0" Y="1" Z="59" Value="-0.8160006" />
        		<Node X="1" Y="1" Z="59" Value="3.976453" />
        		<Node X="2" Y="1" Z="59" Value="7.305612" />
        		<Node X="0" Y="2" Z="59" Value="0.0611175" />
        		<Node X="1" Y="2" Z="59" Value="-0.1232658" />
        		<Node X="2" Y="2" Z="59" Value="-0.6120446" />
        		"""
    grid_movement_x = []
    for line in nodedatatest.strip().split("\n"):
        start = line.find('Value="') + len('Value="')
        end = line.find('"', start)
        value = float(line[start:end])
        grid_movement_x.append(value)
    grid_movement_x = np.asarray(grid_movement_x)
    assert np.array_equal(get_data_from_warp_xml(xml_file_path, "GridMovementX", 2), grid_movement_x)

    #level2, wrong(not correct) node values
    nodedatatest1 = """ 
    <Node X="0" Y="0" Z="0" Value="0" /> ###Wrong value
        		<Node X="1" Y="0" Z="0" Value="5.320158" />
        		<Node X="2" Y="0" Z="0" Value="1.452159" />
        		<Node X="0" Y="1" Z="0" Value="-2.376527" />
        		<Node X="1" Y="1" Z="0" Value="-1.308048" />
        		<Node X="2" Y="1" Z="0" Value="1.455077" />
        		<Node X="0" Y="2" Z="0" Value="0.3901345" />
        		<Node X="1" Y="2" Z="0" Value="-0.5754877" />
        		<Node X="2" Y="2" Z="0" Value="-0.4075638" />
        		<Node X="0" Y="0" Z="1" Value="-0.7106043" />
        		<Node X="1" Y="0" Z="1" Value="5.864784" />
        		<Node X="2" Y="0" Z="1" Value="3.30233" />
        		<Node X="0" Y="1" Z="1" Value="-1.52878" />
        		<Node X="1" Y="1" Z="1" Value="-2.483566" />
        		<Node X="2" Y="1" Z="1" Value="-4.079008" />
        		<Node X="0" Y="2" Z="1" Value="0.2063846" />
        		<Node X="1" Y="2" Z="1" Value="-0.6359377" />
        		<Node X="2" Y="2" Z="1" Value="-0.4039657" />
        		<Node X="0" Y="0" Z="2" Value="-0.4095204" />
        		<Node X="1" Y="0" Z="2" Value="1.397589" />
        		<Node X="2" Y="0" Z="2" Value="0.2903593" />
        		<Node X="0" Y="1" Z="2" Value="0.185078" />
        		<Node X="1" Y="1" Z="2" Value="-1.683118" />
        		<Node X="2" Y="1" Z="2" Value="-2.564194" />
        		<Node X="0" Y="2" Z="2" Value="0.05996355" />
        		<Node X="1" Y="2" Z="2" Value="-0.08808023" />
        		<Node X="2" Y="2" Z="2" Value="0.03305629" />
        		<Node X="0" Y="0" Z="3" Value="-0.5266708" />
        		<Node X="1" Y="0" Z="3" Value="0.4416825" />
        		<Node X="2" Y="0" Z="3" Value="1.395934" />
        		<Node X="0" Y="1" Z="3" Value="-1.08517" />
        		<Node X="1" Y="1" Z="3" Value="-1.230583" />
        		<Node X="2" Y="1" Z="3" Value="-4.273524" />
        		<Node X="0" Y="2" Z="3" Value="0.1407896" />
        		<Node X="1" Y="2" Z="3" Value="0.2722078" />
        		<Node X="2" Y="2" Z="3" Value="0.224728" />
        		<Node X="0" Y="0" Z="4" Value="-0.9587417" />
        		<Node X="1" Y="0" Z="4" Value="-0.2396259" />
        		<Node X="2" Y="0" Z="4" Value="1.009668" />
        		<Node X="0" Y="1" Z="4" Value="-1.290259" />
        		<Node X="1" Y="1" Z="4" Value="-1.248055" />
        		<Node X="2" Y="1" Z="4" Value="-4.197393" />
        		<Node X="0" Y="2" Z="4" Value="0.2185136" />
        		<Node X="1" Y="2" Z="4" Value="-0.2632738" />
        		<Node X="2" Y="2" Z="4" Value="0.06670134" />
        		<Node X="0" Y="0" Z="5" Value="-0.3779781" />
        		<Node X="1" Y="0" Z="5" Value="-1.635345" />
        		<Node X="2" Y="0" Z="5" Value="0.3110349" />
        		<Node X="0" Y="1" Z="5" Value="-0.7143888" />
        		<Node X="1" Y="1" Z="5" Value="-1.022123" />
        		<Node X="2" Y="1" Z="5" Value="-4.777106" />
        		<Node X="0" Y="2" Z="5" Value="0.08590294" />
        		<Node X="1" Y="2" Z="5" Value="0.2265385" />
        		<Node X="2" Y="2" Z="5" Value="0.1821524" />
        		<Node X="0" Y="0" Z="6" Value="-1.530601" />
        		<Node X="1" Y="0" Z="6" Value="-0.5546232" />
        		<Node X="2" Y="0" Z="6" Value="1.850821" />
        		<Node X="0" Y="1" Z="6" Value="-1.653455" />
        		<Node X="1" Y="1" Z="6" Value="-0.7825686" />
        		<Node X="2" Y="1" Z="6" Value="-5.108892" />
        		<Node X="0" Y="2" Z="6" Value="0.3053687" />
        		<Node X="1" Y="2" Z="6" Value="-0.1588862" />
        		<Node X="2" Y="2" Z="6" Value="-0.3030559" />
        		<Node X="0" Y="0" Z="7" Value="-0.3694279" />
        		<Node X="1" Y="0" Z="7" Value="-3.079624" />
        		<Node X="2" Y="0" Z="7" Value="-1.713288" />
        		<Node X="0" Y="1" Z="7" Value="-1.834426" />
        		<Node X="1" Y="1" Z="7" Value="-1.526223" />
        		<Node X="2" Y="1" Z="7" Value="-2.180725" />
        		<Node X="0" Y="2" Z="7" Value="0.1993298" />
        		<Node X="1" Y="2" Z="7" Value="0.3071583" />
        		<Node X="2" Y="2" Z="7" Value="0.3434351" />
        		<Node X="0" Y="0" Z="8" Value="0.7162532" />
        		<Node X="1" Y="0" Z="8" Value="-2.697225" />
        		<Node X="2" Y="0" Z="8" Value="-1.586078" />
        		<Node X="0" Y="1" Z="8" Value="0.321272" />
        		<Node X="1" Y="1" Z="8" Value="-1.675499" />
        		<Node X="2" Y="1" Z="8" Value="-1.831161" />
        		<Node X="0" Y="2" Z="8" Value="-0.1197902" />
        		<Node X="1" Y="2" Z="8" Value="0.3940544" />
        		<Node X="2" Y="2" Z="8" Value="0.3820015" />
        		<Node X="0" Y="0" Z="9" Value="-0.4948481" />
        		<Node X="1" Y="0" Z="9" Value="0.01214749" />
        		<Node X="2" Y="0" Z="9" Value="-1.277677" />
        		<Node X="0" Y="1" Z="9" Value="-2.367667" />
        		<Node X="1" Y="1" Z="9" Value="-2.466241" />
        		<Node X="2" Y="1" Z="9" Value="-2.800977" />
        		<Node X="0" Y="2" Z="9" Value="0.2596546" />
        		<Node X="1" Y="2" Z="9" Value="0.3332871" />
        		<Node X="2" Y="2" Z="9" Value="0.1657559" />
        		<Node X="0" Y="0" Z="10" Value="0.2982151" />
        		<Node X="1" Y="0" Z="10" Value="-1.50164" />
        		<Node X="2" Y="0" Z="10" Value="-1.518515" />
        		<Node X="0" Y="1" Z="10" Value="-0.6211068" />
        		<Node X="1" Y="1" Z="10" Value="-2.937962" />
        		<Node X="2" Y="1" Z="10" Value="-4.469218" />
        		<Node X="0" Y="2" Z="10" Value="0.01148285" />
        		<Node X="1" Y="2" Z="10" Value="0.7165185" />
        		<Node X="2" Y="2" Z="10" Value="0.5784123" />
        		<Node X="0" Y="0" Z="11" Value="-0.5220225" />
        		<Node X="1" Y="0" Z="11" Value="-3.54739" />
        		<Node X="2" Y="0" Z="11" Value="-0.470219" />
        		<Node X="0" Y="1" Z="11" Value="-1.295103" />
        		<Node X="1" Y="1" Z="11" Value="-1.014738" />
        		<Node X="2" Y="1" Z="11" Value="-1.174498" />
        		<Node X="0" Y="2" Z="11" Value="0.1662981" />
        		<Node X="1" Y="2" Z="11" Value="0.4758431" />
        		<Node X="2" Y="2" Z="11" Value="-0.009146304" />
        		<Node X="0" Y="0" Z="12" Value="-1.012112" />
        		<Node X="1" Y="0" Z="12" Value="-3.174333" />
        		<Node X="2" Y="0" Z="12" Value="0.8814076" />
        		<Node X="0" Y="1" Z="12" Value="-1.22106" />
        		<Node X="1" Y="1" Z="12" Value="-1.344784" />
        		<Node X="2" Y="1" Z="12" Value="-3.342522" />
        		<Node X="0" Y="2" Z="12" Value="0.1792351" />
        		<Node X="1" Y="2" Z="12" Value="0.2866584" />
        		<Node X="2" Y="2" Z="12" Value="-0.07647342" />
        		<Node X="0" Y="0" Z="13" Value="-0.4943229" />
        		<Node X="1" Y="0" Z="13" Value="-1.95817" />
        		<Node X="2" Y="0" Z="13" Value="-0.6274569" />
        		<Node X="0" Y="1" Z="13" Value="-1.629542" />
        		<Node X="1" Y="1" Z="13" Value="-1.810308" />
        		<Node X="2" Y="1" Z="13" Value="-2.223524" />
        		<Node X="0" Y="2" Z="13" Value="0.142798" />
        		<Node X="1" Y="2" Z="13" Value="0.5942734" />
        		<Node X="2" Y="2" Z="13" Value="0.3156306" />
        		<Node X="0" Y="0" Z="14" Value="0.1230368" />
        		<Node X="1" Y="0" Z="14" Value="-1.255964" />
        		<Node X="2" Y="0" Z="14" Value="-2.560725" />
        		<Node X="0" Y="1" Z="14" Value="-0.2168628" />
        		<Node X="1" Y="1" Z="14" Value="-2.68361" />
        		<Node X="2" Y="1" Z="14" Value="-2.556492" />
        		<Node X="0" Y="2" Z="14" Value="-0.01003722" />
        		<Node X="1" Y="2" Z="14" Value="0.2361742" />
        		<Node X="2" Y="2" Z="14" Value="0.4976555" />
        		<Node X="0" Y="0" Z="15" Value="-1.611152" />
        		<Node X="1" Y="0" Z="15" Value="-2.151034" />
        		<Node X="2" Y="0" Z="15" Value="-2.108429" />
        		<Node X="0" Y="1" Z="15" Value="-1.768244" />
        		<Node X="1" Y="1" Z="15" Value="-1.228219" />
        		<Node X="2" Y="1" Z="15" Value="-3.732646" />
        		<Node X="0" Y="2" Z="15" Value="0.2576976" />
        		<Node X="1" Y="2" Z="15" Value="0.3873455" />
        		<Node X="2" Y="2" Z="15" Value="0.5294612" />
        		<Node X="0" Y="0" Z="16" Value="-0.6109841" />
        		<Node X="1" Y="0" Z="16" Value="-0.681752" />
        		<Node X="2" Y="0" Z="16" Value="-1.912903" />
        		<Node X="0" Y="1" Z="16" Value="-1.016887" />
        		<Node X="1" Y="1" Z="16" Value="-1.598863" />
        		<Node X="2" Y="1" Z="16" Value="-3.753352" />
        		<Node X="0" Y="2" Z="16" Value="0.1342393" />
        		<Node X="1" Y="2" Z="16" Value="0.2953961" />
        		<Node X="2" Y="2" Z="16" Value="0.2839231" />
        		<Node X="0" Y="0" Z="17" Value="-0.315821" />
        		<Node X="1" Y="0" Z="17" Value="-0.9707435" />
        		<Node X="2" Y="0" Z="17" Value="-0.00040579" />
        		<Node X="0" Y="1" Z="17" Value="-1.032178" />
        		<Node X="1" Y="1" Z="17" Value="-0.2488512" />
        		<Node X="2" Y="1" Z="17" Value="-4.106051" />
        		<Node X="0" Y="2" Z="17" Value="0.1406045" />
        		<Node X="1" Y="2" Z="17" Value="0.1552727" />
        		<Node X="2" Y="2" Z="17" Value="-0.01827077" />
        		<Node X="0" Y="0" Z="18" Value="0.3701774" />
        		<Node X="1" Y="0" Z="18" Value="-1.723352" />
        		<Node X="2" Y="0" Z="18" Value="-1.372155" />
        		<Node X="0" Y="1" Z="18" Value="-0.8536774" />
        		<Node X="1" Y="1" Z="18" Value="-0.56396" />
        		<Node X="2" Y="1" Z="18" Value="-2.582149" />
        		<Node X="0" Y="2" Z="18" Value="0.1191429" />
        		<Node X="1" Y="2" Z="18" Value="0.2179693" />
        		<Node X="2" Y="2" Z="18" Value="-0.07021593" />
        		<Node X="0" Y="0" Z="19" Value="-0.1060382" />
        		<Node X="1" Y="0" Z="19" Value="-1.811532" />
        		<Node X="2" Y="0" Z="19" Value="-1.88311" />
        		<Node X="0" Y="1" Z="19" Value="-0.4577783" />
        		<Node X="1" Y="1" Z="19" Value="-0.09073634" />
        		<Node X="2" Y="1" Z="19" Value="-1.782525" />
        		<Node X="0" Y="2" Z="19" Value="0.06478995" />
        		<Node X="1" Y="2" Z="19" Value="0.2026641" />
        		<Node X="2" Y="2" Z="19" Value="0.09382563" />
        		<Node X="0" Y="0" Z="20" Value="-0.08359869" />
        		<Node X="1" Y="0" Z="20" Value="-0.7033283" />
        		<Node X="2" Y="0" Z="20" Value="-1.705231" />
        		<Node X="0" Y="1" Z="20" Value="-0.3581509" />
        		<Node X="1" Y="1" Z="20" Value="-1.204437" />
        		<Node X="2" Y="1" Z="20" Value="-1.552996" />
        		<Node X="0" Y="2" Z="20" Value="0.03866892" />
        		<Node X="1" Y="2" Z="20" Value="0.06212883" />
        		<Node X="2" Y="2" Z="20" Value="0.1897697" />
        		<Node X="0" Y="0" Z="21" Value="-1.195053" />
        		<Node X="1" Y="0" Z="21" Value="-1.273353" />
        		<Node X="2" Y="0" Z="21" Value="0.1509104" />
        		<Node X="0" Y="1" Z="21" Value="-0.6179267" />
        		<Node X="1" Y="1" Z="21" Value="-0.2443093" />
        		<Node X="2" Y="1" Z="21" Value="-2.65915" />
        		<Node X="0" Y="2" Z="21" Value="0.1130377" />
        		<Node X="1" Y="2" Z="21" Value="-0.0814635" />
        		<Node X="2" Y="2" Z="21" Value="-0.1326908" />
        		<Node X="0" Y="0" Z="22" Value="0.008384475" />
        		<Node X="1" Y="0" Z="22" Value="-1.716682" />
        		<Node X="2" Y="0" Z="22" Value="-1.825313" />
        		<Node X="0" Y="1" Z="22" Value="0.6198087" />
        		<Node X="1" Y="1" Z="22" Value="0.210767" />
        		<Node X="2" Y="1" Z="22" Value="-2.328422" />
        		<Node X="0" Y="2" Z="22" Value="-0.05069068" />
        		<Node X="1" Y="2" Z="22" Value="-0.09321602" />
        		<Node X="2" Y="2" Z="22" Value="-0.009880386" />
        		<Node X="0" Y="0" Z="23" Value="-0.4430967" />
        		<Node X="1" Y="0" Z="23" Value="-1.305517" />
        		<Node X="2" Y="0" Z="23" Value="-1.719509" />
        		<Node X="0" Y="1" Z="23" Value="0.1801253" />
        		<Node X="1" Y="1" Z="23" Value="-0.4872607" />
        		<Node X="2" Y="1" Z="23" Value="-1.736029" />
        		<Node X="0" Y="2" Z="23" Value="-0.01621913" />
        		<Node X="1" Y="2" Z="23" Value="0.04965311" />
        		<Node X="2" Y="2" Z="23" Value="0.1059583" />
        		<Node X="0" Y="0" Z="24" Value="-0.15272" />
        		<Node X="1" Y="0" Z="24" Value="-1.140445" />
        		<Node X="2" Y="0" Z="24" Value="-1.187778" />
        		<Node X="0" Y="1" Z="24" Value="-0.8995698" />
        		<Node X="1" Y="1" Z="24" Value="0.1290679" />
        		<Node X="2" Y="1" Z="24" Value="-2.039777" />
        		<Node X="0" Y="2" Z="24" Value="0.1745232" />
        		<Node X="1" Y="2" Z="24" Value="-0.2871963" />
        		<Node X="2" Y="2" Z="24" Value="-0.1064552" />
        		<Node X="0" Y="0" Z="25" Value="1.473913" />
        		<Node X="1" Y="0" Z="25" Value="-0.6776252" />
        		<Node X="2" Y="0" Z="25" Value="-1.987762" />
        		<Node X="0" Y="1" Z="25" Value="0.2574184" />
        		<Node X="1" Y="1" Z="25" Value="0.1255547" />
        		<Node X="2" Y="1" Z="25" Value="0.2881294" />
        		<Node X="0" Y="2" Z="25" Value="-0.04728567" />
        		<Node X="1" Y="2" Z="25" Value="-0.05159292" />
        		<Node X="2" Y="2" Z="25" Value="-0.11628" />
        		<Node X="0" Y="0" Z="26" Value="0.01599271" />
        		<Node X="1" Y="0" Z="26" Value="-0.463193" />
        		<Node X="2" Y="0" Z="26" Value="-1.147276" />
        		<Node X="0" Y="1" Z="26" Value="2.109198" />
        		<Node X="1" Y="1" Z="26" Value="0.1329704" />
        		<Node X="2" Y="1" Z="26" Value="-1.443648" />
        		<Node X="0" Y="2" Z="26" Value="-0.2384964" />
        		<Node X="1" Y="2" Z="26" Value="-0.1099772" />
        		<Node X="2" Y="2" Z="26" Value="0.3359437" />
        		<Node X="0" Y="0" Z="27" Value="-0.09431294" />
        		<Node X="1" Y="0" Z="27" Value="-0.7216987" />
        		<Node X="2" Y="0" Z="27" Value="-0.4680083" />
        		<Node X="0" Y="1" Z="27" Value="0.6385919" />
        		<Node X="1" Y="1" Z="27" Value="0.2349623" />
        		<Node X="2" Y="1" Z="27" Value="-0.6990215" />
        		<Node X="0" Y="2" Z="27" Value="-0.07671299" />
        		<Node X="1" Y="2" Z="27" Value="-0.1488184" />
        		<Node X="2" Y="2" Z="27" Value="0.1199325" />
        		<Node X="0" Y="0" Z="28" Value="0.06634519" />
        		<Node X="1" Y="0" Z="28" Value="-0.2920912" />
        		<Node X="2" Y="0" Z="28" Value="-0.5589179" />
        		<Node X="0" Y="1" Z="28" Value="1.428783" />
        		<Node X="1" Y="1" Z="28" Value="0.3591155" />
        		<Node X="2" Y="1" Z="28" Value="0.9722123" />
        		<Node X="0" Y="2" Z="28" Value="-0.2322642" />
        		<Node X="1" Y="2" Z="28" Value="0.1640536" />
        		<Node X="2" Y="2" Z="28" Value="0.3133739" />
        		<Node X="0" Y="0" Z="29" Value="0.7822325" />
        		<Node X="1" Y="0" Z="29" Value="0.3300999" />
        		<Node X="2" Y="0" Z="29" Value="-1.627798" />
        		<Node X="0" Y="1" Z="29" Value="-0.1763324" />
        		<Node X="1" Y="1" Z="29" Value="0.6838462" />
        		<Node X="2" Y="1" Z="29" Value="1.724693" />
        		<Node X="0" Y="2" Z="29" Value="-0.00318099" />
        		<Node X="1" Y="2" Z="29" Value="-0.02396142" />
        		<Node X="2" Y="2" Z="29" Value="0.1055321" />
        		<Node X="0" Y="0" Z="30" Value="0.1433068" />
        		<Node X="1" Y="0" Z="30" Value="1.349256" />
        		<Node X="2" Y="0" Z="30" Value="2.26653" />
        		<Node X="0" Y="1" Z="30" Value="1.422405" />
        		<Node X="1" Y="1" Z="30" Value="2.410702" />
        		<Node X="2" Y="1" Z="30" Value="3.195401" />
        		<Node X="0" Y="2" Z="30" Value="-0.2058323" />
        		<Node X="1" Y="2" Z="30" Value="-0.4314051" />
        		<Node X="2" Y="2" Z="30" Value="-0.1900425" />
        		<Node X="0" Y="0" Z="31" Value="-1.179764" />
        		<Node X="1" Y="0" Z="31" Value="0.883917" />
        		<Node X="2" Y="0" Z="31" Value="1.174489" />
        		<Node X="0" Y="1" Z="31" Value="0.09098724" />
        		<Node X="1" Y="1" Z="31" Value="2.660588" />
        		<Node X="2" Y="1" Z="31" Value="2.385371" />
        		<Node X="0" Y="2" Z="31" Value="0.03932115" />
        		<Node X="1" Y="2" Z="31" Value="-0.4811596" />
        		<Node X="2" Y="2" Z="31" Value="-0.1107998" />
        		<Node X="0" Y="0" Z="32" Value="-0.6113384" />
        		<Node X="1" Y="0" Z="32" Value="0.6766121" />
        		<Node X="2" Y="0" Z="32" Value="0.9443513" />
        		<Node X="0" Y="1" Z="32" Value="0.8643097" />
        		<Node X="1" Y="1" Z="32" Value="1.983773" />
        		<Node X="2" Y="1" Z="32" Value="2.509074" />
        		<Node X="0" Y="2" Z="32" Value="-0.1268053" />
        		<Node X="1" Y="2" Z="32" Value="-0.2978492" />
        		<Node X="2" Y="2" Z="32" Value="0.2010989" />
        		<Node X="0" Y="0" Z="33" Value="0.2941595" />
        		<Node X="1" Y="0" Z="33" Value="-0.003231251" />
        		<Node X="2" Y="0" Z="33" Value="0.4881998" />
        		<Node X="0" Y="1" Z="33" Value="1.208933" />
        		<Node X="1" Y="1" Z="33" Value="1.942395" />
        		<Node X="2" Y="1" Z="33" Value="1.338014" />
        		<Node X="0" Y="2" Z="33" Value="-0.1935601" />
        		<Node X="1" Y="2" Z="33" Value="-0.3215258" />
        		<Node X="2" Y="2" Z="33" Value="-0.05696579" />
        		<Node X="0" Y="0" Z="34" Value="-0.3472976" />
        		<Node X="1" Y="0" Z="34" Value="0.09433644" />
        		<Node X="2" Y="0" Z="34" Value="0.3397444" />
        		<Node X="0" Y="1" Z="34" Value="0.127166" />
        		<Node X="1" Y="1" Z="34" Value="1.557775" />
        		<Node X="2" Y="1" Z="34" Value="1.189507" />
        		<Node X="0" Y="2" Z="34" Value="-0.03112097" />
        		<Node X="1" Y="2" Z="34" Value="-0.1582837" />
        		<Node X="2" Y="2" Z="34" Value="0.216153" />
        		<Node X="0" Y="0" Z="35" Value="-0.03777345" />
        		<Node X="1" Y="0" Z="35" Value="-0.4355727" />
        		<Node X="2" Y="0" Z="35" Value="1.36326" />
        		<Node X="0" Y="1" Z="35" Value="0.3529037" />
        		<Node X="1" Y="1" Z="35" Value="1.915063" />
        		<Node X="2" Y="1" Z="35" Value="0.5705025" />
        		<Node X="0" Y="2" Z="35" Value="-0.05825124" />
        		<Node X="1" Y="2" Z="35" Value="-0.1032214" />
        		<Node X="2" Y="2" Z="35" Value="0.3005217" />
        		<Node X="0" Y="0" Z="36" Value="-0.8399256" />
        		<Node X="1" Y="0" Z="36" Value="0.3645122" />
        		<Node X="2" Y="0" Z="36" Value="0.2628777" />
        		<Node X="0" Y="1" Z="36" Value="-0.01787591" />
        		<Node X="1" Y="1" Z="36" Value="1.024429" />
        		<Node X="2" Y="1" Z="36" Value="1.444087" />
        		<Node X="0" Y="2" Z="36" Value="-0.01184748" />
        		<Node X="1" Y="2" Z="36" Value="0.02578417" />
        		<Node X="2" Y="2" Z="36" Value="0.06493462" />
        		<Node X="0" Y="0" Z="37" Value="0.3468409" />
        		<Node X="1" Y="0" Z="37" Value="-0.5145742" />
        		<Node X="2" Y="0" Z="37" Value="0.6361068" />
        		<Node X="0" Y="1" Z="37" Value="1.043736" />
        		<Node X="1" Y="1" Z="37" Value="0.5854191" />
        		<Node X="2" Y="1" Z="37" Value="1.211399" />
        		<Node X="0" Y="2" Z="37" Value="-0.1399718" />
        		<Node X="1" Y="2" Z="37" Value="-0.1381016" />
        		<Node X="2" Y="2" Z="37" Value="-0.1884141" />
        		<Node X="0" Y="0" Z="38" Value="-0.3449529" />
        		<Node X="1" Y="0" Z="38" Value="0.01692064" />
        		<Node X="2" Y="0" Z="38" Value="1.315026" />
        		<Node X="0" Y="1" Z="38" Value="-0.8521376" />
        		<Node X="1" Y="1" Z="38" Value="1.14969" />
        		<Node X="2" Y="1" Z="38" Value="1.065405" />
        		<Node X="0" Y="2" Z="38" Value="0.1287372" />
        		<Node X="1" Y="2" Z="38" Value="-0.2063174" />
        		<Node X="2" Y="2" Z="38" Value="-0.4680429" />
        		<Node X="0" Y="0" Z="39" Value="0.2715719" />
        		<Node X="1" Y="0" Z="39" Value="-0.8855376" />
        		<Node X="2" Y="0" Z="39" Value="3.277173" />
        		<Node X="0" Y="1" Z="39" Value="0.2681186" />
        		<Node X="1" Y="1" Z="39" Value="1.410705" />
        		<Node X="2" Y="1" Z="39" Value="0.6290808" />
        		<Node X="0" Y="2" Z="39" Value="-0.06141897" />
        		<Node X="1" Y="2" Z="39" Value="-0.09191106" />
        		<Node X="2" Y="2" Z="39" Value="-0.04201572" />
        		<Node X="0" Y="0" Z="40" Value="-0.1328561" />
        		<Node X="1" Y="0" Z="40" Value="-0.6741854" />
        		<Node X="2" Y="0" Z="40" Value="0.6705673" />
        		<Node X="0" Y="1" Z="40" Value="-0.525125" />
        		<Node X="1" Y="1" Z="40" Value="1.498924" />
        		<Node X="2" Y="1" Z="40" Value="0.7334902" />
        		<Node X="0" Y="2" Z="40" Value="0.06934692" />
        		<Node X="1" Y="2" Z="40" Value="-0.04914386" />
        		<Node X="2" Y="2" Z="40" Value="-0.4205783" />
        		<Node X="0" Y="0" Z="41" Value="-0.2773003" />
        		<Node X="1" Y="0" Z="41" Value="-0.9362093" />
        		<Node X="2" Y="0" Z="41" Value="-0.2420284" />
        		<Node X="0" Y="1" Z="41" Value="0.0848609" />
        		<Node X="1" Y="1" Z="41" Value="0.489738" />
        		<Node X="2" Y="1" Z="41" Value="1.853848" />
        		<Node X="0" Y="2" Z="41" Value="0.01720244" />
        		<Node X="1" Y="2" Z="41" Value="-0.07237782" />
        		<Node X="2" Y="2" Z="41" Value="-0.4250088" />
        		<Node X="0" Y="0" Z="42" Value="0.5253142" />
        		<Node X="1" Y="0" Z="42" Value="-1.027279" />
        		<Node X="2" Y="0" Z="42" Value="0.1768653" />
        		<Node X="0" Y="1" Z="42" Value="1.107163" />
        		<Node X="1" Y="1" Z="42" Value="1.797172" />
        		<Node X="2" Y="1" Z="42" Value="1.85055" />
        		<Node X="0" Y="2" Z="42" Value="-0.1254745" />
        		<Node X="1" Y="2" Z="42" Value="-0.2504082" />
        		<Node X="2" Y="2" Z="42" Value="-0.2743937" />
        		<Node X="0" Y="0" Z="43" Value="0.1132936" />
        		<Node X="1" Y="0" Z="43" Value="-1.633733" />
        		<Node X="2" Y="0" Z="43" Value="0.5591072" />
        		<Node X="0" Y="1" Z="43" Value="0.7104746" />
        		<Node X="1" Y="1" Z="43" Value="1.830815" />
        		<Node X="2" Y="1" Z="43" Value="1.436787" />
        		<Node X="0" Y="2" Z="43" Value="-0.1145731" />
        		<Node X="1" Y="2" Z="43" Value="0.1250063" />
        		<Node X="2" Y="2" Z="43" Value="-0.157924" />
        		<Node X="0" Y="0" Z="44" Value="-0.349147" />
        		<Node X="1" Y="0" Z="44" Value="0.3923524" />
        		<Node X="2" Y="0" Z="44" Value="-0.2887461" />
        		<Node X="0" Y="1" Z="44" Value="0.1901588" />
        		<Node X="1" Y="1" Z="44" Value="1.04338" />
        		<Node X="2" Y="1" Z="44" Value="1.554292" />
        		<Node X="0" Y="2" Z="44" Value="0.01251187" />
        		<Node X="1" Y="2" Z="44" Value="-0.07875831" />
        		<Node X="2" Y="2" Z="44" Value="-0.3525737" />
        		<Node X="0" Y="0" Z="45" Value="0.2201274" />
        		<Node X="1" Y="0" Z="45" Value="-2.493914" />
        		<Node X="2" Y="0" Z="45" Value="0.7733946" />
        		<Node X="0" Y="1" Z="45" Value="-0.8519823" />
        		<Node X="1" Y="1" Z="45" Value="1.001191" />
        		<Node X="2" Y="1" Z="45" Value="2.241728" />
        		<Node X="0" Y="2" Z="45" Value="0.03651591" />
        		<Node X="1" Y="2" Z="45" Value="0.4484932" />
        		<Node X="2" Y="2" Z="45" Value="-0.06920049" />
        		<Node X="0" Y="0" Z="46" Value="-0.936515" />
        		<Node X="1" Y="0" Z="46" Value="-1.173076" />
        		<Node X="2" Y="0" Z="46" Value="1.238834" />
        		<Node X="0" Y="1" Z="46" Value="0.9181442" />
        		<Node X="1" Y="1" Z="46" Value="1.752505" />
        		<Node X="2" Y="1" Z="46" Value="0.5683882" />
        		<Node X="0" Y="2" Z="46" Value="-0.07035258" />
        		<Node X="1" Y="2" Z="46" Value="0.03659626" />
        		<Node X="2" Y="2" Z="46" Value="-0.02968399" />
        		<Node X="0" Y="0" Z="47" Value="0.1427311" />
        		<Node X="1" Y="0" Z="47" Value="-3.099199" />
        		<Node X="2" Y="0" Z="47" Value="-0.9049471" />
        		<Node X="0" Y="1" Z="47" Value="0.771832" />
        		<Node X="1" Y="1" Z="47" Value="1.965032" />
        		<Node X="2" Y="1" Z="47" Value="1.828468" />
        		<Node X="0" Y="2" Z="47" Value="-0.1411418" />
        		<Node X="1" Y="2" Z="47" Value="0.2229437" />
        		<Node X="2" Y="2" Z="47" Value="0.1171091" />
        		<Node X="0" Y="0" Z="48" Value="0.1909064" />
        		<Node X="1" Y="0" Z="48" Value="0.5024482" />
        		<Node X="2" Y="0" Z="48" Value="1.095386" />
        		<Node X="0" Y="1" Z="48" Value="0.3759831" />
        		<Node X="1" Y="1" Z="48" Value="1.345439" />
        		<Node X="2" Y="1" Z="48" Value="0.6294759" />
        		<Node X="0" Y="2" Z="48" Value="-0.02446398" />
        		<Node X="1" Y="2" Z="48" Value="-0.193266" />
        		<Node X="2" Y="2" Z="48" Value="-0.2045425" />
        		<Node X="0" Y="0" Z="49" Value="0.7162808" />
        		<Node X="1" Y="0" Z="49" Value="-2.214506" />
        		<Node X="2" Y="0" Z="49" Value="0.8649447" />
        		<Node X="0" Y="1" Z="49" Value="1.871275" />
        		<Node X="1" Y="1" Z="49" Value="2.490769" />
        		<Node X="2" Y="1" Z="49" Value="1.605241" />
        		<Node X="0" Y="2" Z="49" Value="-0.2157702" />
        		<Node X="1" Y="2" Z="49" Value="0.0398576" />
        		<Node X="2" Y="2" Z="49" Value="-0.30811" />
        		<Node X="0" Y="0" Z="50" Value="0.538143" />
        		<Node X="1" Y="0" Z="50" Value="-0.8824145" />
        		<Node X="2" Y="0" Z="50" Value="0.01733304" />
        		<Node X="0" Y="1" Z="50" Value="0.04796191" />
        		<Node X="1" Y="1" Z="50" Value="2.129442" />
        		<Node X="2" Y="1" Z="50" Value="2.682483" />
        		<Node X="0" Y="2" Z="50" Value="-0.02275831" />
        		<Node X="1" Y="2" Z="50" Value="-0.329757" />
        		<Node X="2" Y="2" Z="50" Value="-0.4441737" />
        		<Node X="0" Y="0" Z="51" Value="-0.07612614" />
        		<Node X="1" Y="0" Z="51" Value="-2.40694" />
        		<Node X="2" Y="0" Z="51" Value="0.03193626" />
        		<Node X="0" Y="1" Z="51" Value="-0.2932324" />
        		<Node X="1" Y="1" Z="51" Value="2.623614" />
        		<Node X="2" Y="1" Z="51" Value="1.631239" />
        		<Node X="0" Y="2" Z="51" Value="0.0364047" />
        		<Node X="1" Y="2" Z="51" Value="0.1970877" />
        		<Node X="2" Y="2" Z="51" Value="-0.1242779" />
        		<Node X="0" Y="0" Z="52" Value="-0.833669" />
        		<Node X="1" Y="0" Z="52" Value="-1.254515" />
        		<Node X="2" Y="0" Z="52" Value="-0.168173" />
        		<Node X="0" Y="1" Z="52" Value="-2.982485" />
        		<Node X="1" Y="1" Z="52" Value="3.099726" />
        		<Node X="2" Y="1" Z="52" Value="2.312814" />
        		<Node X="0" Y="2" Z="52" Value="0.2808829" />
        		<Node X="1" Y="2" Z="52" Value="0.2039993" />
        		<Node X="2" Y="2" Z="52" Value="-0.0898677" />
        		<Node X="0" Y="0" Z="53" Value="-0.3158733" />
        		<Node X="1" Y="0" Z="53" Value="-2.156559" />
        		<Node X="2" Y="0" Z="53" Value="2.417242" />
        		<Node X="0" Y="1" Z="53" Value="-1.047516" />
        		<Node X="1" Y="1" Z="53" Value="3.627437" />
        		<Node X="2" Y="1" Z="53" Value="1.944811" />
        		<Node X="0" Y="2" Z="53" Value="0.09728091" />
        		<Node X="1" Y="2" Z="53" Value="0.1532321" />
        		<Node X="2" Y="2" Z="53" Value="-0.6192255" />
        		<Node X="0" Y="0" Z="54" Value="-0.04899749" />
        		<Node X="1" Y="0" Z="54" Value="2.848426" />
        		<Node X="2" Y="0" Z="54" Value="0.2904806" />
        		<Node X="0" Y="1" Z="54" Value="-2.252651" />
        		<Node X="1" Y="1" Z="54" Value="2.962822" />
        		<Node X="2" Y="1" Z="54" Value="3.871899" />
        		<Node X="0" Y="2" Z="54" Value="0.1288705" />
        		<Node X="1" Y="2" Z="54" Value="-0.09419243" />
        		<Node X="2" Y="2" Z="54" Value="-0.1377179" />
        		<Node X="0" Y="0" Z="55" Value="0.8474329" />
        		<Node X="1" Y="0" Z="55" Value="1.214333" />
        		<Node X="2" Y="0" Z="55" Value="-0.52802" />
        		<Node X="0" Y="1" Z="55" Value="-0.4807324" />
        		<Node X="1" Y="1" Z="55" Value="4.788553" />
        		<Node X="2" Y="1" Z="55" Value="0.5269803" />
        		<Node X="0" Y="2" Z="55" Value="-0.134818" />
        		<Node X="1" Y="2" Z="55" Value="-0.2000996" />
        		<Node X="2" Y="2" Z="55" Value="0.07953535" />
        		<Node X="0" Y="0" Z="56" Value="0.1961866" />
        		<Node X="1" Y="0" Z="56" Value="0.630663" />
        		<Node X="2" Y="0" Z="56" Value="-1.200837" />
        		<Node X="0" Y="1" Z="56" Value="-1.504695" />
        		<Node X="1" Y="1" Z="56" Value="4.681966" />
        		<Node X="2" Y="1" Z="56" Value="3.99545" />
        		<Node X="0" Y="2" Z="56" Value="0.05390552" />
        		<Node X="1" Y="2" Z="56" Value="-0.2746132" />
        		<Node X="2" Y="2" Z="56" Value="0.09590138" />
        		<Node X="0" Y="0" Z="57" Value="-0.192025" />
        		<Node X="1" Y="0" Z="57" Value="2.120564" />
        		<Node X="2" Y="0" Z="57" Value="-0.5196728" />
        		<Node X="0" Y="1" Z="57" Value="-2.184315" />
        		<Node X="1" Y="1" Z="57" Value="3.931082" />
        		<Node X="2" Y="1" Z="57" Value="4.095085" />
        		<Node X="0" Y="2" Z="57" Value="0.1436728" />
        		<Node X="1" Y="2" Z="57" Value="-0.6218677" />
        		<Node X="2" Y="2" Z="57" Value="0.02701958" />
        		<Node X="0" Y="0" Z="58" Value="0.6309627" />
        		<Node X="1" Y="0" Z="58" Value="-0.869297" />
        		<Node X="2" Y="0" Z="58" Value="0.2767608" />
        		<Node X="0" Y="1" Z="58" Value="1.617473" />
        		<Node X="1" Y="1" Z="58" Value="5.58949" />
        		<Node X="2" Y="1" Z="58" Value="6.701704" />
        		<Node X="0" Y="2" Z="58" Value="-0.2136974" />
        		<Node X="1" Y="2" Z="58" Value="-0.151066" />
        		<Node X="2" Y="2" Z="58" Value="-0.08292479" />
        		<Node X="0" Y="0" Z="59" Value="-0.1033018" />
        		<Node X="1" Y="0" Z="59" Value="-0.8923529" />
        		<Node X="2" Y="0" Z="59" Value="1.542079" />
        		<Node X="0" Y="1" Z="59" Value="-0.8160006" />
        		<Node X="1" Y="1" Z="59" Value="3.976453" />
        		<Node X="2" Y="1" Z="59" Value="7.305612" />
        		<Node X="0" Y="2" Z="59" Value="0.0611175" />
        		<Node X="1" Y="2" Z="59" Value="-0.1232658" />
        		<Node X="2" Y="2" Z="59" Value="-0.6120446" />
    """
    grid_movement_x1 = []
    for line in nodedatatest.strip().split("\n"):
        start = line.find('Value="') + len('Value="')
        end = line.find('"', start)
        value = line[start:end]
        grid_movement_x1.append(value)
    grid_movement_x1 = np.asarray(grid_movement_x1)
    assert not np.array_equal(get_data_from_warp_xml(xml_file_path, "GridMovementX", 2), grid_movement_x1)

    # level 1, negative floats
    axisY_negative = [
    -95.60207, -110.2457, -89.65818, -122.977, -75.29381, -145.2447, -88.36815,
    -134.5136, -79.06438, -144.8848, -80.94444, -148.9504, -98.82204, -131.589,
    -99.3392, -134.0308, -108.0799, -132.8979, -102.9604, -109.8975, -123.7455,
    -156.5575, -71.82225, -128.3785, -141.1216, -111.4745, -105.5518, -107.1509,
    -141.7749, 0, -88.65258, 52.32646, -125.3249, 7.120238, -62.03314, -51.76203,
    -64.6597, -33.80381, -44.38956, -33.72198, -69.23692, 9.592659, -64.28349,
    -26.47904, -21.23565, -0.4442041, -37.51812, 10.85751, -52.15116, 23.32489,
    -40.59274, 20.9565, -7.281012, 32.14969, -19.47281, 54.45569, -12.57366,
    46.69533, -21.36308, 59.44625]
    axisY_negative = np.asarray(axisY_negative)
    assert np.array_equal(get_data_from_warp_xml(xml_file_path, "AxisOffsetY", 1), axisY_negative)

    # level 3, must raise Value Error exception
    with pytest.raises(ValueError):
        get_data_from_warp_xml(xml_file_path, "AxisOffsetY", 3)

    # level1/2 with absent node_name
    assert np.array_equal(get_data_from_warp_xml(xml_file_path, "Absent", 2), None)


    # Level 1 on something that has children, what happens. this function doesn't really handle this possibility
    #get_data_from_warp_xml(xml_file_path, "GridCTF", 1)

    #level 1, strings, (paths) -- fails!
    # Level 1, booleans -- fails because reads 'True' as string but should be interpreted as boolean!
    """
    moviepath = np.array["017_01.mrc","017_02.mrc","017_03.mrc","017_04.mrc","017_05.mrc","017_06.mrc",
    "017_07.mrc","017_08.mrc","017_09.mrc","017_10.mrc","017_11.mrc","017_12.mrc",
    "017_13.mrc","017_14.mrc","017_15.mrc","017_16.mrc","017_17.mrc","017_18.mrc",
    "017_19.mrc","017_20.mrc","017_21.mrc","017_22.mrc","017_23.mrc","017_24.mrc",
    "017_25.mrc","017_26.mrc","017_27.mrc","017_28.mrc","017_29.mrc","017_30.mrc",
    "017_31.mrc","017_32.mrc","017_33.mrc","017_34.mrc","017_35.mrc","017_36.mrc",
    "017_37.mrc","017_38.mrc","017_39.mrc","017_40.mrc","017_41.mrc","017_42.mrc",
    "017_43.mrc","017_44.mrc","017_45.mrc","017_46.mrc","017_47.mrc","017_48.mrc",
    "017_49.mrc","017_50.mrc","017_51.mrc","017_52.mrc","017_53.mrc","017_54.mrc",
    "017_55.mrc","017_56.mrc","017_57.mrc","017_58.mrc","017_59.mrc","017_60.mrc"]
    assert np.array_equal(get_data_from_warp_xml(xml_file_path, "MoviePath", 1), moviepath)
    
    usetilt = np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
       True,  True,  True,  True,  True,  True,  True,  True,  True,
       True,  True,  True,  True,  True,  True,  True,  True,  True,
       True,  True,  True,  True,  True,  True,  True,  True,  True,
       True,  True,  True,  True,  True,  True,  True,  True,  True,
       True,  True,  True,  True,  True,  True,  True,  True,  True,
       True,  True,  True,  True,  True,  True,  True,  True,  True])
    assert np.array_equal(get_data_from_warp_xml(xml_file_path, "UseTilt", 1), usetilt)

    """


def test_warp_ctf_read():
    current_dir = Path(__file__).parent
    xml_file_path = str(current_dir / "test_data" / "TS_018" / "018.xml")
    gridctf  = [
    3.409815, 3.398723, 3.389318, 3.392323, 3.424242, 3.371011, 3.403402, 3.406947, 3.42302, 3.378743,
    3.392365, 3.390212, 3.419454, 3.38763, 3.404047, 3.388702, 3.404345, 3.378603, 3.388518, 3.382968,
    3.370579, 3.384403, 3.392067, 3.387562, 3.397726, 3.391282, 3.387162, 3.37812, 3.374513, 3.382992,
    3.383263, 3.376348, 3.385212, 3.385358, 3.367899, 3.360602, 3.350081, 3.362278, 3.352579, 3.345044,
    3.331407, 3.356446, 3.34499, 3.327942, 3.324675, 3.344646, 3.350497, 3.312646, 3.308363, 3.295593,
    3.296385, 3.273645, 3.300465, 3.2972, 3.288325, 3.248272, 3.270276, 3.309069, 3.392386, 3.353555]
    gridctf = np.asarray(gridctf, dtype=float)


    gridctf_da = [
    164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
    164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
    164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
    164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
    164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
    164, 164, 164, 164, 164, 164, 164, 164, 164, 164]
    gridctf_da = np.asarray(gridctf_da, dtype=float)


    gridctf_phase = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gridctf_phase = np.asarray(gridctf_phase, dtype=float)

    test_columns = ["defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"]
    test_df = pd.DataFrame(columns=test_columns)
    test_df["defocus_mean"] = test_df["defocus1"] = test_df["defocus2"] = gridctf
    test_df["astigmatism"] = gridctf_da
    test_df["phase_shift"] = gridctf_phase

    pd.testing.assert_frame_equal(test_df, warp_ctf_read(xml_file_path))

#something not clear: with star file we have defocus1 and defocus2 and defmean calculated by these ,
#with warp xml we have just the mean?
#Astigmatism is not converted to micrometers
#no check of micrometers for any column
def test_gtcf_read():
    current_dir = Path(__file__).parent
    #018_absent_phaseShift18
    defocusU_test18 = [35268.359375, 34332.652344, 33858.722656, 35649.660156, 35397.246094, 34354.187500, 35621.707031, 34577.019531, 34172.589844, 35716.156250, 34074.867188, 33485.125000, 33248.859375, 33834.765625, 34547.640625, 34658.406250, 34407.492188, 34153.625000, 33948.054688, 34577.507812, 33752.703125, 34121.878906, 33979.281250, 34352.917969, 33891.410156, 34112.527344, 33924.605469, 33581.265625, 33641.425781, 33717.992188, 33825.421875, 33744.929688, 34003.250000, 33772.292969, 33350.425781, 33389.335938, 33201.953125, 33414.808594, 33217.554688, 33301.515625, 33487.601562, 32610.074219, 32928.492188, 33161.558594, 32418.701172, 32699.968750, 34008.625000, 32632.125000, 32065.232422, 33369.921875, 32399.195312, 31207.765625, 32922.281250, 32578.910156, 32438.275391, 32354.269531, 33756.363281, 33170.699219, 37010.957031, 32093.152344
]
    defocusV_test18 = [35036.406250, 35647.824219, 33047.644531, 35557.472656, 35075.472656, 34716.179688, 36280.660156, 35192.824219, 35397.730469, 35151.062500, 34475.976562, 34870.453125, 36184.812500, 34223.851562, 34803.976562, 34864.046875, 34018.304688, 34377.671875, 34938.710938, 33981.203125, 34219.953125, 34331.261719, 34223.937500, 33963.550781, 34128.144531, 33965.644531, 34102.753906, 33985.140625, 33995.378906, 34121.898438, 33678.492188, 33899.648438, 33789.945312, 34020.761719, 33809.808594, 33860.757812, 33309.765625, 33522.691406, 33591.093750, 33596.937500, 32750.677734, 33761.066406, 33548.140625, 32592.550781, 33085.218750, 33276.828125, 32096.939453, 32348.343750, 32434.767578, 32137.892578, 33612.585938, 32769.007812, 31738.017578, 31882.125000, 31464.076172, 33423.136719, 32141.511719, 32602.736328, 36586.699219, 33375.855469
]
    defocusAngle18 = [21.260185, 10.495987, 82.230835, 83.206253, 55.838074, 51.147915, 49.745102, 62.932869, 63.413452, 75.131836, 67.024940, 31.398960, 51.957737, 42.132729, 0.877556, 75.411850, 19.592590, 61.788765, 17.310394, 2.286995, 67.911240, 73.547852, 10.120667, 81.420303, 47.649994, 70.209831, 76.795242, 83.089111, 64.373100, 71.908905, 64.891617, 88.230179, 17.646332, 64.991364, 40.967464, 62.539276, 47.918793, 24.099361, 7.947800, 34.639038, 59.527859, 21.130569, 78.265640, 9.343567, 26.686031, 53.716324, 59.336975, 84.182068, 11.137737, 8.331212, 55.481678, 59.722404, 43.838737, 37.230209, 24.560413, 44.282932, 60.271088, 46.200169, 39.174423, 75.284645
]
    phase_shift18 =   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]
    defocusU_test18 = np.asarray(defocusU_test18, dtype=float)
    defocusV_test18 = np.asarray(defocusV_test18, dtype=float)
    defocusAngle18 = np.asarray(defocusAngle18, dtype=float)
    defocusU_test18 = defocusU_test18 * 10e-5
    defocusV_test18 = defocusV_test18 * 10e-5
    #defocusAngle_18 = defocusAngle_18 * 10e-5 ASTIGMATISM IS NOT IN MICROMETERS!?
    phase_shift18 = np.asarray(phase_shift18, dtype=float)
    defocus_mean= (defocusU_test18 + defocusV_test18)/2


    df_test = pd.DataFrame(
        {
            'defocus1': defocusU_test18,
            'defocus2': defocusV_test18,
            'astigmatism': defocusAngle18,
            'phase_shift': phase_shift18,
            'defocus_mean': defocus_mean

        }
    )
    pd.testing.assert_frame_equal(df_test, gctf_read(str(current_dir / "test_data" / "TS_018" / "018_gctf.star")))

def test_cttfind4_read():
    current_dir = Path(__file__).parent

    defocus1_test18 = [14477.870117, 18332.730469, 19371.796875, 18046.099609, 18457.386719, 17372.529297, 18232.728516, 18506.312500, 18353.582031, 19893.322266, 18975.546875, 18963.015625, 18881.785156, 19096.888672, 19281.589844, 18903.167969, 19031.861328, 19032.248047, 18853.355469, 18952.986328, 18920.476562, 19107.906250, 19206.066406, 19276.335938, 19355.328125, 19192.632812, 19025.808594, 19359.962891, 19148.015625, 19321.939453, 19337.650391, 18820.208984, 18972.789062, 19654.128906, 19007.716797, 18973.917969, 19478.619141, 19328.347656, 19642.240234, 19599.916016, 19806.958984, 18332.730469, 19371.796875, 18046.099609, 18457.386719, 17372.529297, 18232.728516, 18506.312500, 18353.582031, 19893.322266, 18975.546875, 18963.015625, 18881.785156, 19096.888672, 19281.589844, 18903.167969, 19031.861328, 19032.248047, 18853.355469, 18853.355469]
    defocus2_test18 = [13500.001953, 17896.296875, 18665.488281, 17408.392578, 17909.847656, 17371.917969, 17712.291016, 18223.146484, 17973.222656, 19385.732422, 18861.855469, 18590.837891, 18429.767578, 18741.103516, 18953.857422, 18632.029297, 18947.216797, 18861.468750, 18658.119141, 18654.927734, 18812.078125, 18737.226562, 19152.679688, 18986.986328, 19108.808594, 19052.113281, 18965.224609, 18877.849609, 18812.537109, 18919.740234, 18981.781250, 18593.011719, 18727.517578, 19254.160156, 18051.062500, 18802.392578, 19049.289062, 18845.232422, 19373.451172, 19060.380859, 19740.335938, 17896.296875, 18665.488281, 17408.392578, 17909.847656, 17371.917969, 17712.291016, 18223.146484, 17973.222656, 19385.732422, 18861.855469, 18590.837891, 18429.767578, 18741.103516, 18953.857422, 18632.029297, 18947.216797, 18861.468750, 18658.119141, 22658.119141]
    defocusAngle_test18 = [75.000012, -84.116628, 39.985953, -87.499983, 12.500009, -48.154572, -28.166397, -11.545737, 59.680465, 21.019584, -41.021307, -39.846347, 51.624298, 51.607670, 22.806290, -66.451669, -71.873917, -80.792067, 32.230116, 78.747006, 79.855710, 81.304823, 12.261773, 74.979433, -89.945750, 28.671587, -49.862850, 65.000008, -44.448914, 85.094567, -89.727498, 44.394792, -15.041164, -77.267396, -51.400244, 82.015853, -11.604279, -71.148598, -85.041511, 27.514029, -70.902534, -84.116628, 39.985953, -87.499983, 12.500009, -48.154572, -28.166397, -11.545737, 59.680465, 21.019584, -41.021307, -39.846347, 51.624298, 51.607670, 22.806290, -66.451669, -71.873917, -80.792067, 32.230116, 32.230116]
    phase_shift_test18 = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

    defocus1_test18 = np.asarray(defocus1_test18, dtype=np.float32)
    defocus2_test18 = np.asarray(defocus2_test18, dtype=np.float32)
    defocusAngle_test18 = np.asarray(defocusAngle_test18, dtype=np.float32)
    phase_shift_test18 = np.asarray(phase_shift_test18, dtype=np.float32)
    defocus1_test18 *= 10e-5 #micrometers
    defocus2_test18 *= 10e-5 #micrometers
    defocus_mean_test18 = (defocus1_test18 + defocus2_test18) /2

    df_test18 = pd.DataFrame(
        {
            'defocus1': defocus1_test18,
            'defocus2': defocus2_test18,
            'astigmatism': defocusAngle_test18,
            'phase_shift': phase_shift_test18,
            'defocus_mean': defocus_mean_test18
        }
    )
    pd.testing.assert_frame_equal(df_test18, ctffind4_read(str(current_dir / "test_data" / "TS_018" / "018_ctffind4.txt")))

#Why to assume pandas df is correct? Contains correct amount of columns, contains the correct amount of data per each column
#Test if defocus values are in micrometers, phase shift in radians?
#Something not clear: why do we get different defocus values from each different file format? xml, star, ctffind4
def test_defocus_load():
    #File1
    current_dir = Path(__file__).parent

    gridctf = [
        3.409815, 3.398723, 3.389318, 3.392323, 3.424242, 3.371011, 3.403402, 3.406947, 3.42302, 3.378743,
        3.392365, 3.390212, 3.419454, 3.38763, 3.404047, 3.388702, 3.404345, 3.378603, 3.388518, 3.382968,
        3.370579, 3.384403, 3.392067, 3.387562, 3.397726, 3.391282, 3.387162, 3.37812, 3.374513, 3.382992,
        3.383263, 3.376348, 3.385212, 3.385358, 3.367899, 3.360602, 3.350081, 3.362278, 3.352579, 3.345044,
        3.331407, 3.356446, 3.34499, 3.327942, 3.324675, 3.344646, 3.350497, 3.312646, 3.308363, 3.295593,
        3.296385, 3.273645, 3.300465, 3.2972, 3.288325, 3.248272, 3.270276, 3.309069, 3.392386, 3.353555]
    gridctf = np.asarray(gridctf, dtype=float)

    gridctf_da = [
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
        164, 164, 164, 164, 164, 164, 164, 164, 164, 164]
    gridctf_da = np.asarray(gridctf_da, dtype=float)

    gridctf_phase = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gridctf_phase = np.asarray(gridctf_phase, dtype=float)

    test_columns = ["defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"]
    test_df_xml = pd.DataFrame(columns=test_columns)
    test_df_xml["defocus_mean"] = test_df_xml["defocus1"] = test_df_xml["defocus2"] = gridctf
    test_df_xml["astigmatism"] = gridctf_da
    test_df_xml["phase_shift"] = gridctf_phase


    #Test defocus_load from warp xml file
    pd.testing.assert_frame_equal(test_df_xml, defocus_load(str(current_dir / "test_data" / "TS_018" / "018.xml"), "warp"))


    defocusU_test18 = [35268.359375, 34332.652344, 33858.722656, 35649.660156, 35397.246094, 34354.187500, 35621.707031,
                       34577.019531, 34172.589844, 35716.156250, 34074.867188, 33485.125000, 33248.859375, 33834.765625,
                       34547.640625, 34658.406250, 34407.492188, 34153.625000, 33948.054688, 34577.507812, 33752.703125,
                       34121.878906, 33979.281250, 34352.917969, 33891.410156, 34112.527344, 33924.605469, 33581.265625,
                       33641.425781, 33717.992188, 33825.421875, 33744.929688, 34003.250000, 33772.292969, 33350.425781,
                       33389.335938, 33201.953125, 33414.808594, 33217.554688, 33301.515625, 33487.601562, 32610.074219,
                       32928.492188, 33161.558594, 32418.701172, 32699.968750, 34008.625000, 32632.125000, 32065.232422,
                       33369.921875, 32399.195312, 31207.765625, 32922.281250, 32578.910156, 32438.275391, 32354.269531,
                       33756.363281, 33170.699219, 37010.957031, 32093.152344
                       ]
    defocusV_test18 = [35036.406250, 35647.824219, 33047.644531, 35557.472656, 35075.472656, 34716.179688, 36280.660156,
                       35192.824219, 35397.730469, 35151.062500, 34475.976562, 34870.453125, 36184.812500, 34223.851562,
                       34803.976562, 34864.046875, 34018.304688, 34377.671875, 34938.710938, 33981.203125, 34219.953125,
                       34331.261719, 34223.937500, 33963.550781, 34128.144531, 33965.644531, 34102.753906, 33985.140625,
                       33995.378906, 34121.898438, 33678.492188, 33899.648438, 33789.945312, 34020.761719, 33809.808594,
                       33860.757812, 33309.765625, 33522.691406, 33591.093750, 33596.937500, 32750.677734, 33761.066406,
                       33548.140625, 32592.550781, 33085.218750, 33276.828125, 32096.939453, 32348.343750, 32434.767578,
                       32137.892578, 33612.585938, 32769.007812, 31738.017578, 31882.125000, 31464.076172, 33423.136719,
                       32141.511719, 32602.736328, 36586.699219, 33375.855469
                       ]
    defocusAngle18 = [21.260185, 10.495987, 82.230835, 83.206253, 55.838074, 51.147915, 49.745102, 62.932869, 63.413452,
                      75.131836, 67.024940, 31.398960, 51.957737, 42.132729, 0.877556, 75.411850, 19.592590, 61.788765,
                      17.310394, 2.286995, 67.911240, 73.547852, 10.120667, 81.420303, 47.649994, 70.209831, 76.795242,
                      83.089111, 64.373100, 71.908905, 64.891617, 88.230179, 17.646332, 64.991364, 40.967464, 62.539276,
                      47.918793, 24.099361, 7.947800, 34.639038, 59.527859, 21.130569, 78.265640, 9.343567, 26.686031,
                      53.716324, 59.336975, 84.182068, 11.137737, 8.331212, 55.481678, 59.722404, 43.838737, 37.230209,
                      24.560413, 44.282932, 60.271088, 46.200169, 39.174423, 75.284645
                      ]
    phase_shift18 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                     ]
    defocusU_test18 = np.asarray(defocusU_test18, dtype=float)
    defocusV_test18 = np.asarray(defocusV_test18, dtype=float)
    defocusAngle18 = np.asarray(defocusAngle18, dtype=float)
    defocusU_test18 = defocusU_test18 * 10e-5
    defocusV_test18 = defocusV_test18 * 10e-5
    # defocusAngle_18 = defocusAngle_18 * 10e-5 ASTIGMATISM IS NOT IN MICROMETERS!?
    phase_shift18 = np.asarray(phase_shift18, dtype=float)
    defocus_mean = (defocusU_test18 + defocusV_test18) / 2

    df_test_star = pd.DataFrame(
        {
            'defocus1': defocusU_test18,
            'defocus2': defocusV_test18,
            'astigmatism': defocusAngle18,
            'phase_shift': phase_shift18,
            'defocus_mean': defocus_mean

        }
    )
    # Test defocus_load from gctf_star file
    pd.testing.assert_frame_equal(df_test_star, defocus_load(str(current_dir / "test_data" / "TS_018" / "018_gctf.star"), "gctf"))

    defocus1_test18 = [14477.870117, 18332.730469, 19371.796875, 18046.099609, 18457.386719, 17372.529297, 18232.728516,
                       18506.312500, 18353.582031, 19893.322266, 18975.546875, 18963.015625, 18881.785156, 19096.888672,
                       19281.589844, 18903.167969, 19031.861328, 19032.248047, 18853.355469, 18952.986328, 18920.476562,
                       19107.906250, 19206.066406, 19276.335938, 19355.328125, 19192.632812, 19025.808594, 19359.962891,
                       19148.015625, 19321.939453, 19337.650391, 18820.208984, 18972.789062, 19654.128906, 19007.716797,
                       18973.917969, 19478.619141, 19328.347656, 19642.240234, 19599.916016, 19806.958984, 18332.730469,
                       19371.796875, 18046.099609, 18457.386719, 17372.529297, 18232.728516, 18506.312500, 18353.582031,
                       19893.322266, 18975.546875, 18963.015625, 18881.785156, 19096.888672, 19281.589844, 18903.167969,
                       19031.861328, 19032.248047, 18853.355469, 18853.355469]
    defocus2_test18 = [13500.001953, 17896.296875, 18665.488281, 17408.392578, 17909.847656, 17371.917969, 17712.291016,
                       18223.146484, 17973.222656, 19385.732422, 18861.855469, 18590.837891, 18429.767578, 18741.103516,
                       18953.857422, 18632.029297, 18947.216797, 18861.468750, 18658.119141, 18654.927734, 18812.078125,
                       18737.226562, 19152.679688, 18986.986328, 19108.808594, 19052.113281, 18965.224609, 18877.849609,
                       18812.537109, 18919.740234, 18981.781250, 18593.011719, 18727.517578, 19254.160156, 18051.062500,
                       18802.392578, 19049.289062, 18845.232422, 19373.451172, 19060.380859, 19740.335938, 17896.296875,
                       18665.488281, 17408.392578, 17909.847656, 17371.917969, 17712.291016, 18223.146484, 17973.222656,
                       19385.732422, 18861.855469, 18590.837891, 18429.767578, 18741.103516, 18953.857422, 18632.029297,
                       18947.216797, 18861.468750, 18658.119141, 22658.119141]
    defocusAngle_test18 = [75.000012, -84.116628, 39.985953, -87.499983, 12.500009, -48.154572, -28.166397, -11.545737,
                           59.680465, 21.019584, -41.021307, -39.846347, 51.624298, 51.607670, 22.806290, -66.451669,
                           -71.873917, -80.792067, 32.230116, 78.747006, 79.855710, 81.304823, 12.261773, 74.979433,
                           -89.945750, 28.671587, -49.862850, 65.000008, -44.448914, 85.094567, -89.727498, 44.394792,
                           -15.041164, -77.267396, -51.400244, 82.015853, -11.604279, -71.148598, -85.041511, 27.514029,
                           -70.902534, -84.116628, 39.985953, -87.499983, 12.500009, -48.154572, -28.166397, -11.545737,
                           59.680465, 21.019584, -41.021307, -39.846347, 51.624298, 51.607670, 22.806290, -66.451669,
                           -71.873917, -80.792067, 32.230116, 32.230116]
    phase_shift_test18 = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

    defocus1_test18 = np.asarray(defocus1_test18, dtype=np.float32)
    defocus2_test18 = np.asarray(defocus2_test18, dtype=np.float32)
    defocusAngle_test18 = np.asarray(defocusAngle_test18, dtype=np.float32)
    phase_shift_test18 = np.asarray(phase_shift_test18, dtype=np.float32)
    defocus1_test18 *= 10e-5  # micrometers
    defocus2_test18 *= 10e-5  # micrometers
    defocus_mean_test18 = (defocus1_test18 + defocus2_test18) / 2

    df_test18_ctffind4 = pd.DataFrame(
        {
            'defocus1': defocus1_test18,
            'defocus2': defocus2_test18,
            'astigmatism': defocusAngle_test18,
            'phase_shift': phase_shift_test18,
            'defocus_mean': defocus_mean_test18
        }
    )
    #Test defocus_load from ctffind file
    pd.testing.assert_frame_equal(df_test18_ctffind4,
                                  defocus_load(str(current_dir / "test_data" / "TS_018" / "018_ctffind4.txt"), "ctffind4"))

    #Test defocus_load passing a pd dataframe with any extension file
    pd.testing.assert_frame_equal(df_test18_ctffind4,
                                  defocus_load(df_test18_ctffind4, "ctffind4"))
    pd.testing.assert_frame_equal(df_test18_ctffind4,
                                  defocus_load(df_test18_ctffind4, "warp"))
    pd.testing.assert_frame_equal(df_test18_ctffind4,
                                  defocus_load(df_test18_ctffind4, "gctf"))
    #random extension, shouldn't affect it, meaningless
    pd.testing.assert_frame_equal(df_test18_ctffind4,
                                  defocus_load(df_test18_ctffind4, "random"))

    #Passing numpy array
    # Create a 2D numpy array with shape (N, 5), filled with random values
    numpy_array = np.random.random((10, 5)) * 100
    # Create a DataFrame with column names for clarity
    df_numpy = pd.DataFrame(numpy_array, columns=["defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"])
    pd.testing.assert_frame_equal(df_numpy,
                                  defocus_load(numpy_array, "random"))

    #Test if exception is being raised correctly
    file_type_test = "random"
    with pytest.raises(Exception):
        output_exception = defocus_load("string", "file_type_test")


def test_one_value_per_line_read():
    data = "1.0\n2.0\n3.0\n"
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(data.encode('utf-8'))
        tmpfile_path = tmpfile.name
    result = one_value_per_line_read(tmpfile_path, data_type=np.float32)
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    data = "1\n2\n3\n"
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(data.encode('utf-8'))
        tmpfile_path = tmpfile.name
    result = one_value_per_line_read(tmpfile_path, data_type=np.int32)
    expected = np.array([1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
    data = ""
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(data.encode('utf-8'))
        tmpfile_path = tmpfile.name
    with pytest.raises(ValueError, match="The input file is empty or contains no valid data."):
        one_value_per_line_read(tmpfile_path)
    data = "1.0\nnot_a_number\n3.0\n"
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(data.encode('utf-8'))
        tmpfile_path = tmpfile.name
    with pytest.raises(ValueError):
        one_value_per_line_read(tmpfile_path, data_type=np.float32)
    os.remove(tmpfile_path)


def create_test_csv_dose(correctedDose=None, removed=None):
    # Define the target directory and CSV file path
    current_dir = Path(__file__).parent / "test_data" / "TS_018"
    file_path = current_dir / "dose_test.csv"
    """if file_path.exists():
        #print(f"CSV file already exists at: {file_path}")
        return file_path"""
    #current_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    # Data to be written to the CSV file
    if removed is None:
        data = {'CorrectedDose' : correctedDose}
        columns = ['CorrectedDose']
    elif correctedDose is None:
        data = {'Removed' : removed}
        columns = ['CorrectedDose']
    else:
        data = {'CorrectedDose': correctedDose,
            'Removed': removed}
        columns = ['CorrectedDose', 'Removed']

    # Create a pandas DataFrame and save it as a CSV file
    df = pd.DataFrame(data)
    df.to_csv(
        file_path,
        index=True,  # write the index, why it has to be true
        float_format="%.1f",  # Format floats with 1 decimal place
        sep=",",  # Use a comma as the delimiter
        header=True, # Write the header row
        #columns=columns
    )
    return file_path
#MDOC to write
def test_total_dose_load():
    #Input is ndarray, should be equal to return
    input_ndarray = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(total_dose_load(input_ndarray), input_ndarray)

    #xml
    #mdoc
    #one value per line (txt)

    #csv
    #Testing removed for 1entry
    filepath = str(create_test_csv_dose([10.0,20.0,30.0,40.0], [False,False,True,False]))
    result = total_dose_load(filepath)
    expected_result = np.array([10.0, 20.0, 40.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected_result)
    #Testing no removed column
    filepath = str(create_test_csv_dose([10.0,20.0,30.0,40.0], None))
    result = total_dose_load(filepath)
    expected_result= np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected_result)
    #Testing exception raise
    #No correctedDose column
    with pytest.raises(ValueError):
        result = total_dose_load(str(create_test_csv_dose(None, [False,False,False,False])))
    if os.path.exists(filepath):
        os.remove(filepath)

    #paths4files
    current_dir = Path(__file__).parent
    mdoc018_path = str(current_dir / "test_data" / "TS_018" / "018.mdoc")
    xml018_path = str(current_dir / "test_data" / "TS_018" / "018.xml")
    txt018_path = str(current_dir / "test_data" / "TS_018" / "018_corrected_dose.txt")

    # Mdoc
    #result = total_dose_load(mdoc018_path, False)
    #to continue

    #xml
    doseXml = np.array([134.3059, 132.0675, 127.5906, 125.3522, 118.6369, 116.3985,
                     109.6832, 107.4447, 100.7294, 98.49098, 91.77568, 89.53725,
                     82.82195, 80.58351, 73.86821, 71.62978, 64.91447, 62.67604,
                     55.96075, 53.72232, 47.00703, 44.7686, 38.05331, 35.81488,
                     29.09959, 26.86116, 20.14587, 17.90744, 11.19215, 8.95372,
                     2.23843, 4.47686, 6.71529, 13.43058, 15.66901, 22.3843,
                     24.62273, 31.33802, 33.57645, 40.29174, 42.53017, 49.24546,
                     51.48389, 58.19918, 60.43761, 67.15291, 69.39134, 76.10664,
                     78.34508, 85.06038, 87.29881, 94.01411, 96.25255, 102.9678,
                     105.2063, 111.9216, 114.16, 120.8753, 123.1138, 129.8291])
    np.testing.assert_array_equal(doseXml, total_dose_load(xml018_path))

    #onevalueperlineFile
    dosetxt = np.array([136.62, 134.38, 127.66, 125.42, 118.71, 116.47, 109.76, 107.52, 100.8, 98.564, 91.848, 89.61, 82.895, 80.656, 73.941, 71.703, 64.987, 62.749, 56.034, 53.795, 47.08, 44.841, 38.126, 35.888, 29.172, 26.934, 20.219, 17.98, 11.265, 9.0266, 2.3113, 4.5497, 6.7881, 13.503, 15.742, 22.457, 24.696, 31.411, 33.649, 40.365, 42.603, 49.318, 51.557, 58.272, 60.51, 67.226, 69.464, 76.179, 78.418, 85.133, 87.372, 94.087, 96.325, 103.04, 105.28, 111.99, 114.23, 120.95, 123.19, 129.9],
                       dtype=np.float32)
    np.testing.assert_array_equal(dosetxt, total_dose_load(txt018_path))

    #not a valid path, not nd array
    with pytest.raises(Exception):
        result = total_dose_load(str("randomString"))

def create_test_csv_angles(angles, order):
    current_dir = Path(__file__).parent / "test_data" / "TS_018"
    file_path = current_dir / "angles_test.csv"
    if order == "zzx": #reversed
        data = {'phi': angles[0], 'psi':angles[2], 'theta':angles[1]}
    elif order == "zxz": #standard
        data = {'phi': angles[0], 'theta':angles[1], 'psi':angles[2]}
    elif order == "exception":
        data = {'phi': angles[0], 'theta':angles[1]}
    df = pd.DataFrame(data)
    df.to_csv(
        file_path,
        index=False,  # write the index, why it has to be true
        float_format="%.1f",  # Format floats with 1 decimal place
        sep=",",  # Use a comma as the delimiter
        header=False,  # Write the header row
        # columns=columns
    )
    return file_path
def test_rot_angles_load():
    #4 x 3 (N x (phi,theta,psi))
    angles_phi_theta_psi = np.array([[1,4,30,45],[2,5,60,45],[3,6,90,120]])

    # N lines: 4 arrays
    result_phi_theta_psi = np.array([[1,2,3],[4,5,6],[30,60,90],[45,45,120]])
    result_phi_psi_theta = np.array([[1,3,2],[4,6,5],[30,90,60],[45,120,45]])


    #Passing nparray: result is equal to input
    # zxz or zzx doesn't matter when passing a numpy array
    np.testing.assert_array_equal(angles_phi_theta_psi, rot_angles_load(angles_phi_theta_psi))

    #testing csv path
    #he lines in csv:
        # phi, theta, psi
        # phi1, theta1, psi1
    filepath = create_test_csv_angles(angles_phi_theta_psi, "zxz")
    np.testing.assert_array_equal(rot_angles_load(str(filepath), "zxz"), result_phi_theta_psi)
    #the lines in csv: must be the same as before: we want this by our will
    filepath = create_test_csv_angles(angles_phi_theta_psi, "zzx")
    np.testing.assert_array_equal(rot_angles_load(str(filepath), "zzx"), result_phi_theta_psi)
    #Testing exception being raised if not valid path is being passed
    with pytest.raises(ValueError):
        rot_angles_load("random")

    #Testing exception being raised if passed csv doesn't contain 3 columns (phi,theta,psi)
    with pytest.raises(ValueError):
        filepath = create_test_csv_angles(angles_phi_theta_psi, "exception")
        result = rot_angles_load((str(filepath), "zxz"))

    #Test file cleanup
    if os.path.exists(filepath):
        os.remove(filepath)
    if os.path.exists(Path(__file__).parent / "test_data" / "TS_018" / "angles_test.csv"):
        os.remove(Path(__file__).parent / "test_data" / "TS_018" / "angles_test.csv")


def test_tlt_load():
    current_dir = Path(__file__).parent / "test_data" / "TS_018"
    angles = [-54, -50, -48, -46, -44, -42, -40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 69
                     ]
    #Passing np should return np as is
    np.testing.assert_equal(np.array(angles), tlt_load(np.array(angles)))

    #Passing list should return np array of list
    np.testing.assert_equal(np.array(angles), tlt_load(angles))

    #Test mdoc
    file_path = current_dir / "018.mdoc"
    angles_mdoc018 = [
        -52.0064, -50.0066, -48.0064, -46.0066, -44.0058, -42.0051, -40.0053, -38.0066, -36.0048, -34.0061, -32.0043,
        -30.0041, -28.0043, -26.0045, -24.0058, -22.005, -20.0043, -18.004, -16.0063, -14.005, -12.0068, -10.007,
        -8.01075, -6.0075, -4.00624, -2.00799, -0.0112337, 1.99052, 3.98928, 5.99103, 7.97729, 9.98604, 11.9893,
        13.9901, 15.9863, 17.9886, 19.9878, 21.9911, 23.9898, 25.9861, 27.9918, 29.9871, 31.9868, 33.9896, 35.9839,
        37.9831, 39.9829, 41.9841, 43.9854, 45.9826, 47.9899, 49.9886, 51.9879, 53.9827, 55.9889, 57.9837, 59.9894,
        61.9822, 63.9869, 65.9732
    ]
    angles_mdoc018 = np.array(angles_mdoc018)
    #sort=False
    np.testing.assert_equal(angles_mdoc018, tlt_load(str(file_path), False))
    #sort=True
    angles_mdoc018_notSortedByDefault = np.copy(angles_mdoc018)
    np.random.shuffle(angles_mdoc018_notSortedByDefault)
    np.testing.assert_equal(np.sort(angles_mdoc018_notSortedByDefault), tlt_load(str(file_path), True))

    #Test xml warp
    angles_xml018 = np.array(angles)
    file_path = current_dir / "018.xml"
    np.testing.assert_equal(angles_xml018, tlt_load(str(file_path), False))
    #sort=True
    angles_xml018_notSortedByDefault = np.copy(angles_xml018)
    np.random.shuffle(angles_xml018_notSortedByDefault)
    np.testing.assert_equal(np.sort(angles_xml018_notSortedByDefault), tlt_load(str(file_path), True))

    #Test onevalueperline file (tlt)
    angles_tlt = np.array([-54.01, -50.01, -48.01, -46.01, -44.01, -42.01, -40.01, -38.01, -36.00, -34.01, -32.00, -30.00, -28.00, -26.00, -24.01, -22.01, -20.00, -18.00, -16.01, -14.01, -12.01, -10.01, -8.01, -6.01, -4.01, -2.01, -0.01, 1.99, 3.99, 5.99, 7.98, 9.99, 11.99, 13.99, 15.99, 17.99, 19.99, 21.99, 23.99, 25.99, 27.99, 29.99, 31.99, 33.99, 35.98, 37.98, 39.98, 41.98, 43.99, 45.98, 47.99, 49.99, 51.99, 53.98, 55.99, 57.98, 59.99, 61.98, 63.99, 69.97

    ], dtype=np.float32)
    file_path = current_dir / "018.tlt"
    angles_tlt_notSortedByDefault = np.copy(angles_tlt)
    np.random.shuffle(angles_tlt_notSortedByDefault)
    #sort=False
    check1 = tlt_load(str(file_path), False)
    np.testing.assert_equal(angles_tlt, check1)
    #sort=True
    np.testing.assert_equal(np.sort(angles_tlt_notSortedByDefault), tlt_load(str(file_path), True))



    #Test raise exceptions
    file_path = current_dir / "random"
    with pytest.raises(ValueError):
        tlt_load(str(file_path)) #not existing file

    with pytest.raises(ValueError):
        tlt_load(32) #invalid input format

    #passing an empty nparray should raise an exception
    with pytest.raises(ValueError):
        tlt_load(np.asarray([]))
    with pytest.raises(ValueError):
        tlt_load([])

def test_dict_write():
    current_dir = Path(__file__).parent / "test_data" / "TS_018"
    file_path = current_dir / "dict_test.json"
    #example dictionary
    dictionary1 = {
        "person1": {
            "name": "Alice",
            "age": 30,
            "skills": ["Python", "Data Analysis"],
            "is_student": False
        },
        "person2": {
            "name": "Bob",
            "age": 25,
            "skills": ["JavaScript", "React"],
            "is_student": True
        }
    }
    try:
        dict_write(dictionary1, file_path)
        assert file_path.exists()
        with open(file_path, "r") as f:
            loaded_data = json.load(f)
        assert loaded_data == dictionary1
    finally: #remove test file
        if file_path.exists():
            file_path.unlink()


def test_dict_load():
    dictionary1 = {
        "person1": {
            "name": "Alice",
            "age": 30,
            "skills": ["Python", "Data Analysis"],
            "is_student": False
        },
        "person2": {
            "name": "Bob",
            "age": 25,
            "skills": ["JavaScript", "React"],
            "is_student": True
        }
    }
    #Passing dict should return dict as is
    assert dictionary1 == dict_load(dictionary1)
    #Passing json string should return dictionary
    assert dictionary1 == dict_load(json.dumps(dictionary1))
    #Testing exception being raised
    with pytest.raises(ValueError):
        invalid_input = 12345
        result = dict_load(invalid_input)
"""
What is the intended behaviour: We are passing to the function an array-like of ints, pointing numbers of lines to be removed;
we are passing an array-like of strings, so that all lines that start with these strings are removed or are kept
What's the point of "ignoring" lines starting with some strings and then passing array-like of numbers of lines to be removed

We make those lines that start with some strings be ignored, and we pass a list of 
integers that are numbers of lines; so if we ignore those lines indexes are going to be "shifted", is this the intended
behaviour? If so then it's working.    
"""

def test_remove_lines():
    # Create a temporary file for testing
    test_file = Path(__file__).parent / "test_data" / "TS_018"/ "remove_lines_test.txt"
    # Sample content to write to the file
    content = """Line 1
Line 2
Line 3
Line 2
Line 4
Test 1
Line 5
"""

    # Write the sample content to the file
    with open(test_file, "w") as f:
        f.write(content)
    # Test case 1: Remove specific lines (by number)
    lines_to_remove = [1, 3]  # We want to remove lines 2 and 4 (remember line numbers are 0-indexed)
    result = remove_lines(str(test_file), lines_to_remove, number_start=0)
    # Check the result
    assert result == ["Line 1\n", "Line 3\n", "Line 4\n", "Test 1\n", "Line 5\n"]

    with open(test_file, "w") as f:
        f.write(content)
    # Test case 2: Skip lines starting with a certain string
    lines_to_remove = [1, 3]  # We want to remove lines 1 and 3
    start_str_to_skip = "Line 2"  # Skip lines starting with "Line 2"
    result = remove_lines(str(test_file), lines_to_remove, start_str_to_skip=start_str_to_skip, number_start=0)

    # Check the result
    assert result == ["Line 1\n", "Line 2\n", "Line 2\n", "Line 4\n", "Line 5\n"]

    with open(test_file, "w") as f:
        f.write(content)
    # Test case 3: Check if output file is written
    output_file = Path(__file__).parent / "test_data" / "TS_018" / "output_file.txt"
    lines_to_remove = 0  # Remove first line
    result = remove_lines(str(test_file), lines_to_remove, number_start=0, output_file=str(output_file))
    # Verify if the output file was created
    assert output_file.exists()
    # Read the content of the output file
    with open(output_file, "r") as f:
        output_content = f.readlines()
    # Check if the output content is as expected
    assert output_content == ["Line 2\n", "Line 3\n", "Line 2\n", "Line 4\n", "Test 1\n", "Line 5\n"]
    # Clean up the created files
    output_file.unlink()
    test_file.unlink()

    if os.path.exists(test_file):
        os.remove(test_file)

#Function is not handling incorrect formatted files?
def test_imod_com_read():
    test_file = Path(__file__).parent / "test_data" / "TS_018" / "tilt.com"
    tilt018_com_expected = {
    "InputProjections": ["018.ali"],
    "OutputFile": ["018.rec"],
    "IMAGEBINNED": [1],
    "TILTFILE": ["018.tlt"],
    "THICKNESS": [1800],
    "RADIAL": [0.35, 0.035],
    "FalloffIsTrueSigma": [1],
    "XAXISTILT": [0.0],
    "LOG": [0.0],
    "SCALE": [0.0, 250.0],
    "PERPENDICULAR": [],
    "Mode": [2],
    "FULLIMAGE": [4096, 4096],
    "SUBSETSTART": [0, 0],
    "AdjustOrigin": [],
    "OFFSET": [0.0],
    "SHIFT": [0.0, 100.0]
    }
    assert tilt018_com_expected == imod_com_read(str(test_file))


@pytest.fixture
def csv_file():
    # Create a temporary file with CSV content
    csv_data = "1 100.0\n2 200.0\n"
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix=".txt") as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name
    yield temp_file_path
    # Cleanup after test
    os.remove(temp_file_path)
def test_z_shift_load(csv_file):
    #case 1: Input is a single number
    result = z_shift_load(100)
    expected_result = pd.DataFrame({"z_shift": [100]})
    pd.testing.assert_frame_equal(result, expected_result)

    #case 2: Input a .com file
    test_file = Path(__file__).parent / "test_data" / "TS_018" / "tilt.com"
    result = z_shift_load(str(test_file))
    expected_result = pd.DataFrame({"z_shift": [100.0]})
    pd.testing.assert_frame_equal(result, expected_result)

    #case3: input is a pd with two columns
    df_input = pd.DataFrame({"tomo_id": [1, 2], "z_shift": [50.0, 75.0]})
    result = z_shift_load(df_input)
    pd.testing.assert_frame_equal(result, df_input)

    #case4: pd with more than 2 columns:
    df_input = pd.DataFrame({"tomo_id": [1,2], "z_shift":[50.0,55.0], "error":["not","valid"]})
    with pytest.raises(ValueError):
        result = z_shift_load(df_input)

    #case5: input is not an int, not a float, not a pd, but a list with two columns
    input_shift = [[1, 50.0], [2, 75.0]]
    expected_result = pd.DataFrame({"tomo_id": [1, 2], "z_shift": [50.0, 75.0]})
    pd.testing.assert_frame_equal(z_shift_load(input_shift), expected_result)
    #We correctly get two columns (tomo_id, s_shift) pd dataframe

    #case6: input is a string but the file doesn't exist
    input_shift = "exit"
    with pytest.raises(ValueError):
        z_shift_load(input_shift)
    #case7: input is not a com file but a valid file
    result = z_shift_load(csv_file)
    expected_result = pd.DataFrame({
        "tomo_id": [1.0, 2.0],
        "z_shift": [100.0, 200.0]
    })
    pd.testing.assert_frame_equal(result, expected_result)

@pytest.fixture
def csv_file2():
    # Create a temporary file with CSV content
    csv_data = "24.0 33.0 -2.3"
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', suffix=".txt") as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name
    yield temp_file_path
    # Cleanup after test
    os.remove(temp_file_path)
#If passing a pd df, it's not creating a new pd dataframe but modifying the input one, intended behaviour?
#If using tomo_idx, tomo_idx column is added as last while others have it as first?
#Mostly inputs only allow 1x3 dimensions
def test_dimensions_load(csv_file2):
    #pd as input 1x3
    pd_input = pd.DataFrame([[1,2,3]], columns=["a", "b", "c"])
    #print(pd_input.columns)
    result = dimensions_load(pd_input) #dimensions_load is already modifying input dataframe
    #print(pd_input.columns)
    #pd_input_test_fail = pd.DataFrame([[1,2,3]], columns=["a", "b", "c"])
    #pd.testing.assert_frame_equal(result, pd_input_test_fail)
    pd_input_test = pd.DataFrame([[1,2,3]], columns=["x", "y", "z"])
    pd.testing.assert_frame_equal(result, pd_input_test)

    #pd as input Nx4 (3x4)
    pd_input1 = pd.DataFrame([[0,1,2,3],[1,1.1,2.1,3.1],[2,1.2,2.2,3.2]])
    pd_input_test1 = pd.DataFrame([[0,1,2,3],[1,1.1,2.1,3.1],[2,1.2,2.2,3.2]], columns=["tomo_id", "x", "y", "z"])
    result1 = dimensions_load(pd_input1)
    pd.testing.assert_frame_equal(result1, pd_input_test1)

    #pd to raise exception (5 columns) shape error
    pd_input2 = pd.DataFrame([[0,1,2,3,4],[1,1,1,1,1]])
    with pytest.raises(ValueError):
        result2 = dimensions_load(pd_input2)

    #com file as input (pathfile)
    test_file = Path(__file__).parent / "test_data" / "TS_018" / "tilt.com"
    expected_result = pd.DataFrame([[4096, 4096, 1800]], columns=["x","y","z"], dtype=np.float64)
    result3 = dimensions_load(str(test_file))
    pd.testing.assert_frame_equal(result3, expected_result)

    #com file with tomo_idx!= null
    #tomo_idx basically works like this: we repeat x,y,z for all tomo_ids in tomo_idx! always same values!

    angles_mdoc018_test2values = [
        -52.0064, -50.0066
    ]
    expected_result = pd.DataFrame([[1,2,3,-52], [1,2,3,-50]], columns=["x", "y", "z", "tomo_id"])
    pd_input4 = pd.DataFrame([[1,2,3]], columns=["x","y","z"])
    result4 = dimensions_load(pd_input4,angles_mdoc018_test2values)
    pd.testing.assert_frame_equal(result4,expected_result)

    #string, not .com file, but csv
    expected_result = pd.DataFrame([[24.0,33.0,-2.3]], columns=["x", "y", "z"])
    pd.testing.assert_frame_equal(dimensions_load(csv_file2), expected_result)

    #string, not existing file
    with pytest.raises(ValueError):
        result = dimensions_load("random_path")

    #test input is list
    test_list = [1,2,3]
    test_list_pd = pd.DataFrame([[1,2,3]], columns=["x", "y", "z"])
    pd.testing.assert_frame_equal(dimensions_load(test_list), test_list_pd)
    #test input is np array
    np_test = np.asarray([1,2,3])
    test_np_pd = pd.DataFrame([[1,2,3]], columns=["x","y","z"])
    pd.testing.assert_frame_equal(dimensions_load(np_test),test_np_pd)
    #test input is np array with other dimensions
    test_np1_pd = pd.DataFrame([[1,2,3,4]], columns=["tomo_id", "x", "y", "z"])
    np1_test = np.asarray([[1,2,3,4]])
    pd.testing.assert_frame_equal(dimensions_load(np1_test),test_np1_pd)

def test_indices_load(tmp_path):
    # Case 0: CSV file with "ToBeRemoved" column
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "ToBeRemoved": [True, False, True, False],
        "OtherColumn": [10, 20, 30, 40]
    })
    df.to_csv(csv_path, index=False)
    indices = indices_load(str(csv_path))
    expected = np.array([0, 2])  # Indices with "ToBeRemoved=True"
    assert np.array_equal(indices, expected), "Failed on CSV file with 'ToBeRemoved' column"

    # Case 1: CSV file with both "Removed" and "ToBeRemoved" columns
    csv_path_with_removed = tmp_path / "test_removed.csv"
    df_with_removed = pd.DataFrame({
        "Removed": [True, False, False, False],
        "ToBeRemoved": [True, True, False, True],
        "OtherColumn": [50, 60, 70, 80]
    })
    df_with_removed.to_csv(csv_path_with_removed, index=False)
    indices = indices_load(str(csv_path_with_removed))
    expected = np.array([1, 3])  # Only rows where "Removed=False" and "ToBeRemoved=True"
    assert np.array_equal(indices, expected), "Failed on CSV file with both 'Removed' and 'ToBeRemoved' columns"


    # Case 2: Text file with one index per line
    text_path = tmp_path / "test.txt"
    text_data = "1\n3\n5\n"
    text_path.write_text(text_data)
    indices = indices_load(str(text_path))
    expected = np.array([0, 2, 4])  # Adjusted to zero-based indexing
    assert np.array_equal(indices, expected), "Failed on text file input"

    # Case 3: List input
    input_list = [1, 3, 5]
    indices = indices_load(input_list)
    expected = np.array([0, 2, 4])  # Adjusted to zero-based indexing
    assert np.array_equal(indices, expected), "Failed on list input"

    # Case 4: Numpy array input
    input_array = np.array([2, 4, 6])
    indices = indices_load(input_array)
    expected = np.array([1, 3, 5])  # Adjusted to zero-based indexing
    assert np.array_equal(indices, expected), "Failed on numpy array input"

    # Case 5: Invalid input type
    with pytest.raises(ValueError, match="Input data must be either path to a valid file either list/array"):
        indices_load(123)  # Invalid input type

    # Case 6: numbered_from_1=False
    indices = indices_load(input_list, numbered_from_1=False)
    expected = np.array([1, 3, 5])  # No adjustment to indexing
    assert np.array_equal(indices, expected), "Failed when numbered_from_1=False"
def test_indices_reset(tmp_path):
    # Create a temporary CSV file
    csv_path = tmp_path / "test_indices_reset.csv"
    df = pd.DataFrame({
        "Removed": [False, False, True, False],
        "ToBeRemoved": [True, False, True, False],
        "OtherColumn": [10, 20, 30, 40]
    })
    df.to_csv(csv_path, index=False)

    # Call the function
    indices_reset(str(csv_path))

    # Read the modified CSV
    modified_df = pd.read_csv(csv_path)

    # Expected DataFrame after applying indices_reset
    expected_df = pd.DataFrame({
        "Removed": [True, False, True, False],  # Updated where ToBeRemoved=True
        "ToBeRemoved": [False, False, False, False],  # Reset to False
        "OtherColumn": [10, 20, 30, 40]
    })

    # Assert that the DataFrame matches the expected result
    pd.testing.assert_frame_equal(modified_df, expected_df, check_dtype=True,
                                   obj="DataFrame after indices_reset")

    # Case: Input file without "Removed" column
    csv_path_no_removed = tmp_path / "test_no_removed.csv"
    df_no_removed = pd.DataFrame({
        "ToBeRemoved": [True, False, True, False],
        "OtherColumn": [10, 20, 30, 40]
    })
    df_no_removed.to_csv(csv_path_no_removed, index=False)

    # Call the function
    indices_reset(str(csv_path_no_removed))

    # Read the modified CSV
    modified_df_no_removed = pd.read_csv(csv_path_no_removed)

    # Expected DataFrame for the case without "Removed" column
    expected_df_no_removed = pd.DataFrame({
        "ToBeRemoved": [False, False, False, False],  # Reset to False
        "OtherColumn": [10, 20, 30, 40]
    })

    # Assert that the DataFrame matches the expected result
    pd.testing.assert_frame_equal(modified_df_no_removed, expected_df_no_removed, check_dtype=True,
                                   obj="DataFrame after indices_reset (no 'Removed' column)")

    # Case: Invalid input path
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        indices_reset("nonexistent_file.csv")

def normalize_lines(file_content):
    return [" ".join(line.split()) for line in file_content.strip().split("\n") if line.strip()]
def test_defocus_remove_file_entries(tmp_path):
    # Creazione del file di input GCTF
    input_file_path = tmp_path / "test_gctf.star"
    input_lines = [
        "data_\n",
        "loop_\n",
        "_rlnMicrographName #1\n",
        "_rlnDefocusU #2\n",
        "file1.mrc 10000.0\n",
        "file2.mrc 11000.0\n",
        "file3.mrc 12000.0\n",
        "file4.mrc 13000.0\n",
    ]
    input_file_path.write_text("".join(input_lines))

    # Elementi da rimuovere: linee 2 e 4 (indice 1-based)
    entries_to_remove = [2, 4]  # Rimuovere "file2.mrc" e "file4.mrc"
    output_file_path = tmp_path / "test_gctf_output.star"

    # Chiamata alla funzione
    defocus_remove_file_entries(
        input_file=str(input_file_path),
        entries_to_remove=entries_to_remove,
        file_type="gctf",
        numbered_from_1=True,
        output_file=str(output_file_path),
    )

    # Contenuto atteso
    expected_lines = [
        "data_",
        "loop_",
        "_rlnMicrographName #1",
        "_rlnDefocusU #2",
        "file1.mrc 10000.0",
        "file3.mrc 12000.0",
    ]

    # Confronto normalizzato
    produced_content = output_file_path.read_text()
    produced_lines = normalize_lines(produced_content)
    expected_normalized = normalize_lines("\n".join(expected_lines))

    assert produced_lines == expected_normalized, "Failed GCTF file case"

    # Case 2: CTFFIND4 file
    input_file_path_ctffind4 = tmp_path / "test_ctffind4.txt"
    input_lines_ctffind4 = [
        "# Comment line\n",
        "file1.ctf\n",
        "file2.ctf\n",
        "file3.ctf\n",
        "file4.ctf\n",
    ]
    input_file_path_ctffind4.write_text("".join(input_lines_ctffind4))

    # Entries to remove: line 2 and 4 (1-based indexing)
    entries_to_remove_ctffind4 = [2, 4]  # Removing "file2.ctf" and "file4.ctf"
    output_file_path_ctffind4 = tmp_path / "test_ctffind4_output.txt"

    defocus_remove_file_entries(
        input_file=str(input_file_path_ctffind4),
        entries_to_remove=entries_to_remove_ctffind4,
        file_type="ctffind4",
        numbered_from_1=True,
        output_file=str(output_file_path_ctffind4),
    )

    # Expected output: keep lines 0, 2, 4 (zero-based)
    expected_lines_ctffind4 = [
        "# Comment line\n",
        "file1.ctf\n",
        "file3.ctf\n",
    ]
    assert output_file_path_ctffind4.read_text() == "".join(expected_lines_ctffind4), "Failed CTFFIND4 file case"

    # Case 3: real star file input


def create_files_in_directory(directory, files):
    os.makedirs(directory, exist_ok=True)
    for file in files:
        with open(os.path.join(directory, file), "w") as f:
            f.write("Test content")
@pytest.fixture
def setup_temp_directory(tmp_path):
    """Fixture to set up a temporary directory with some files."""
    test_dir = tmp_path / "test_files"
    test_files = [
        "file1.txt",
        "file2.txt",
        "image1.png",
        "image2.png",
        "file3.doc",
        "file4_123.doc",
        "file4_abc.doc",
        "file5_456.doc",
    ]
    create_files_in_directory(test_dir, test_files)
    return test_dir, test_files

def test_get_all_files_matching_pattern(setup_temp_directory):
    test_dir, _ = setup_temp_directory

    # Test case 1: Match all .txt files
    pattern = os.path.join(str(test_dir), "*.txt")
    result = get_all_files_matching_pattern(pattern, return_wildcards=False)
    expected = [os.path.join(test_dir, "file1.txt"), os.path.join(test_dir, "file2.txt")]
    assert sorted(result) == sorted(expected)

    # Test case 2: Match all .doc files with wildcards
    pattern = os.path.join(str(test_dir), "*.doc")
    file_names, wildcards = get_all_files_matching_pattern(pattern, numeric_wildcards_only=False,return_wildcards=True)
    expected_files = [
        os.path.join(test_dir, "file3.doc"),
        os.path.join(test_dir, "file4_123.doc"),
        os.path.join(test_dir, "file4_abc.doc"),
        os.path.join(test_dir, "file5_456.doc"),
    ]
    expected_wildcards = ["file3", "file4_abc", "file4_123", "file5_456"]
    assert sorted(file_names) == sorted(expected_files)
    assert sorted(wildcards) == sorted(expected_wildcards)

    # Test case 3: Match files with numeric wildcards only
    pattern = os.path.join(str(test_dir), "file4_*.doc")
    file_names, wildcards = get_all_files_matching_pattern(
        pattern, numeric_wildcards_only=True, return_wildcards=True
    )
    expected_files = [os.path.join(test_dir, "file4_123.doc")]
    expected_wildcards = ["123"]
    assert sorted(file_names) == sorted(expected_files)
    assert sorted(wildcards) == sorted(expected_wildcards)

    #test case: Match files not only with numeric wildcards
    pattern = os.path.join(str(test_dir), "file4_*.doc")
    file_names, wildcards = get_all_files_matching_pattern(
        pattern, numeric_wildcards_only=False, return_wildcards=True
    )
    expected_files = [os.path.join(test_dir, "file4_123.doc"),
                      os.path.join(test_dir, "file4_abc.doc")]
    expected_wildcards = ["abc","123"]
    assert sorted(file_names) == sorted(expected_files)
    assert sorted(wildcards) == sorted(expected_wildcards)

    # Test case 4: No wildcard match for a single file
    pattern = os.path.join(str(test_dir), "file1.txt")
    result = get_all_files_matching_pattern(pattern, return_wildcards=False)
    expected = [os.path.join(test_dir, "file1.txt")]
    assert result == expected

    # Test case 5: Directory does not exist
    with pytest.raises(FileNotFoundError):
        get_all_files_matching_pattern("nonexistent_dir/*.txt")

    # Test case 6: Empty directory
    empty_dir = test_dir / "empty"
    os.makedirs(empty_dir)
    pattern = os.path.join(str(empty_dir), "*.txt")
    result = get_all_files_matching_pattern(pattern, return_wildcards=False)
    assert result == []


def test_sort_files_by_idx():
    #passing a string as an index that can't be converted to integer
    with pytest.raises(ValueError, match="idx_list can't contain elements that can't be converted to integer."):
        res = sort_files_by_idx(["string1", "string2", "string3"], ['1', '2', 'ciao'])

    #passing a string as an index that represents an integer but "outbounds" the file_list array
    with pytest.raises(ValueError, match="idx_list contains invalid indices"):
        res = sort_files_by_idx(["string1", "string2", "string3"],['1','2','4'])

    #passing not a list of strings but a string as file_list
    with pytest.raises(ValueError, match="file_list must be a list of strings"):
        res = sort_files_by_idx("test", ['1'])

    #passing 1 element
    assert sort_files_by_idx(["test.txt"], ["1"]) == ["test.txt"]

    #passing 2 element, with repetition of index, should raise exception
    with pytest.raises(ValueError, match="idx_list contains invalid indices"):
        assert np.array_equal(sort_files_by_idx(["test.txt", "test2.txt"],['1','1']), np.array(["test.txt", "test2.txt"]))

    # passing 2 elements, with two indexes, should produce a regular result
    assert np.array_equal(sort_files_by_idx(["test.txt", "test2.txt"], ['2', '1']), np.array(["test2.txt", "test.txt"]))

    #check if exception is being raised for order
    with pytest.raises(ValueError):
        res = sort_files_by_idx(["test"], ['1'], "randomOrder")

    #check if order is working
    assert np.array_equal(sort_files_by_idx(["test1.txt","test2.txt"],['1','2'], "descending"), np.array(["test2.txt", "test1.txt"]))

    #passing empty list of indexes or file_list is not allowed.
    with pytest.raises(ValueError):
        res = sort_files_by_idx(["file.txt"], [])
    with pytest.raises(ValueError):
        res = sort_files_by_idx([],['1'])

