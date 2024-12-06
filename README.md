# inverse-Mel
Set of adapted files from the Librosa package and DCASE 2020 Autoencoder Baseline system. These files are used for construction of inverse-Mel scale spectrograms.

The following files from the Librosa package should be replaced.
.\core\convert.py
.\filters.py
.\feature\spectral.py

Parameters “isHTK” and “isInverseMel” are passed as arguments in melspectrogram, mel, and mel frequencies functions of the above Librosa package files. 


Adaptation DCASE baseline system.
.\common.py
Concatenation and Average of Mel and inverse-Mel spectrograms are done in the ‘common.py’ file of the DCASE baseline system.
