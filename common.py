"""
 @file   common.py
 @brief  Commonly used script
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os

# additional
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2020 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)

    """    
    mel_spectrogram,mel_f = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    """    

    """    
    """    
    mel_spectrogram,mel_f = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power,
                                                     htk=True, 
                                                     imel=False)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    inv_mel_spectrogram,Imel_f = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power,
                                                     htk=True, 
                                                     imel=True)

    # 03 convert melspectrogram to log mel energy
    inv_log_mel_spectrogram = 20.0 / power * numpy.log10(inv_mel_spectrogram + sys.float_info.epsilon)

    """    
    # Section of code for computation of concatenate spectrogram
    LenSpec = len(log_mel_spectrogram)
    mel_spect_dB_temp_lower = log_mel_spectrogram[0:int(LenSpec/4),:]
    mel_spect_dB_temp_upper = numpy.zeros((int(numpy.shape(log_mel_spectrogram)[0]/4), int(numpy.shape(log_mel_spectrogram)[1])))
    for i in range(int(LenSpec/4),LenSpec-int(LenSpec/4),2):
        mel_spect_dB_temp_upper[int((i-int(LenSpec/4))/2)] = numpy.maximum([log_mel_spectrogram[i,:]],[log_mel_spectrogram[i+1,:]])
    
    iMel_spect_dB_temp_lower = numpy.zeros((int(numpy.shape(inv_log_mel_spectrogram)[0]/4), int(numpy.shape(inv_log_mel_spectrogram)[1])))
    count = 0
    for i in range(int(LenSpec/4),LenSpec-int(LenSpec/4),2):
        iMel_spect_dB_temp_lower[count] = numpy.maximum([inv_log_mel_spectrogram[i,:]],[inv_log_mel_spectrogram[i+1,:]])
        count+=1
    iMel_spect_dB_temp_upper = inv_log_mel_spectrogram[int(LenSpec/4*3):LenSpec,:]
    
    sumMels = numpy.concatenate([mel_spect_dB_temp_lower,mel_spect_dB_temp_upper,iMel_spect_dB_temp_lower, iMel_spect_dB_temp_upper])

    """    
    AvgMels = (log_mel_spectrogram+inv_log_mel_spectrogram)/2

    # print(log_mel_spectrogram[10][10])
    # print(inv_log_mel_spectrogram[10][10])
    # print(AvgMels[10][10])


    # 04 calculate total vector size
    # vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    # vector_array_size = len(inv_log_mel_spectrogram[0, :]) - frames + 1
    # vector_array_size = len(sumMels[0, :]) - frames + 1
    vector_array_size = len(AvgMels[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        # vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
        # vector_array[:, n_mels * t: n_mels * (t + 1)] = inv_log_mel_spectrogram[:, t: t + vector_array_size].T
        # vector_array[:, n_mels * t: n_mels * (t + 1)] = sumMels[:, t: t + vector_array_size].T
        vector_array[:, n_mels * t: n_mels * (t + 1)] = AvgMels[:, t: t + vector_array_size].T

    return vector_array


# load dataset
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        logger.info("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))
    return dirs

########################################################################

