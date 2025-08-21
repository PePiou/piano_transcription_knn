import librosa
import numpy as np
from pathlib import Path
import csv
import os

DATA = Path("piano_notes")     # dossier où se trouvent les fichiers .aiff avec les enregistrements de notes*
FILES = sorted(DATA.glob("*.ai*"))
CSV_OUT  = "Piano_notes_train.csv"

SR      = 44100                # fréquence d’échantillonnage (Hz)
N_FFT   = 2048                 # taille FFT
HOP     = 256 

import numpy as np
import librosa

SR    = 44100
N_FFT = 2048
HOP   = 256

RMS_REL_THRESH = 0.10             # seuil relatif: on garde les fenêtres dont le RMS ≥ 10% du max

def audio_to_vec(path: str) -> np.ndarray:
    """Audio mono -> vecteur 88 dim"""
    y, _ = librosa.load(path, sr=SR, mono=True)
    M = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) 

    rms = librosa.feature.rms(S=M)[0]        
    thr = RMS_REL_THRESH * rms.max()           

    S_sum = np.zeros(M.shape[0], dtype=M.dtype)  
    count = 0
    for i, r in enumerate(rms):               
        if r >= thr:
            S_sum += M[:, i]                  
            count += 1

    S_mean = (S_sum / count) if count > 0 else M.mean(axis=1)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    vec = np.zeros(88, dtype=np.float32)
    for mag, f in zip(S_mean, freqs):
        if f <= 0:                     # on ignore le bin DC
            continue
        midi = int(round(69 + 12*np.log2(f/440.0)))
        if 21 <= midi <= 108:
            vec[midi - 21] += float(mag)
    tot = vec.sum()
    if tot > 0:
        vec /= tot
    return vec

pc_map = {
    'A0': 21, 'A#0': 22, 'Bb0': 22, 'B0': 23,
    'C1': 24, 'C#1': 25, 'Db1': 25, 'D1': 26, 'D#1': 27, 'Eb1': 27, 'E1': 28, 'F1': 29, 'F#1': 30, 'Gb1': 30, 'G1': 31, 'G#1': 32, 'Ab1': 32, 'A1': 33, 'A#1': 34, 'Bb1': 34, 'B1': 35,
    'C2': 36, 'C#2': 37, 'Db2': 37, 'D2': 38, 'D#2': 39, 'Eb2': 39, 'E2': 40, 'F2': 41, 'F#2': 42, 'Gb2': 42, 'G2': 43, 'G#2': 44, 'Ab2': 44, 'A2': 45, 'A#2': 46, 'Bb2': 46, 'B2': 47,
    'C3': 48, 'C#3': 49, 'Db3': 49, 'D3': 50, 'D#3': 51, 'Eb3': 51, 'E3': 52, 'F3': 53, 'F#3': 54, 'Gb3': 54, 'G3': 55, 'G#3': 56, 'Ab3': 56, 'A3': 57, 'A#3': 58, 'Bb3': 58, 'B3': 59,
    'C4': 60, 'C#4': 61, 'Db4': 61, 'D4': 62, 'D#4': 63, 'Eb4': 63, 'E4': 64, 'F4': 65, 'F#4': 66, 'Gb4': 66, 'G4': 67, 'G#4': 68, 'Ab4': 68, 'A4': 69, 'A#4': 70, 'Bb4': 70, 'B4': 71,
    'C5': 72, 'C#5': 73, 'Db5': 73, 'D5': 74, 'D#5': 75, 'Eb5': 75, 'E5': 76, 'F5': 77, 'F#5': 78, 'Gb5': 78, 'G5': 79, 'G#5': 80, 'Ab5': 80, 'A5': 81, 'A#5': 82, 'Bb5': 82, 'B5': 83,
    'C6': 84, 'C#6': 85, 'Db6': 85, 'D6': 86, 'D#6': 87, 'Eb6': 87, 'E6': 88, 'F6': 89, 'F#6': 90, 'Gb6': 90, 'G6': 91, 'G#6': 92, 'Ab6': 92, 'A6': 93, 'A#6': 94, 'Bb6': 94, 'B6': 95,
    'C7': 96, 'C#7': 97, 'Db7': 97, 'D7': 98, 'D#7': 99, 'Eb7': 99, 'E7': 100, 'F7': 101, 'F#7': 102, 'Gb7': 102, 'G7': 103, 'G#7': 104, 'Ab7': 104, 'A7': 105, 'A#7': 106, 'Bb7': 106, 'B7': 107,
    'C8': 108,}

def name_to_midi(filepath: str) -> int:
    """Extrait le numéro MIDI du nom de fichier"""
    name = os.path.basename(filepath)          # Piano.mf.C4.aiff
    stem, _ = os.path.splitext(name)   
    note = stem.split('.')[-1]             # -> "C4"
    note = note.replace('♯', '#').replace('♭', 'b').strip()
    try:
        return pc_map[note]
    except KeyError:
        raise ValueError(f"Note inconnue dans le nom '{name}' ")

with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"feat_{i}" for i in range(88)] + ["midi"])
    for p in FILES:
        vec = audio_to_vec(str(p))          # np.array de longueur 88
        midi = name_to_midi(str(p))         # entier 21..108
        w.writerow(vec.tolist() + [midi])   # on écrit les 88 valeurs + le label

print(f"OK -> {CSV_OUT}")
