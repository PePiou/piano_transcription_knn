import glob
import librosa
import numpy as np
import csv
import os
import tqdm
MAPS_DIR = "MAPS"       # dossier où se trouvent les sous-dossiers ISOL_*
PATTERN  = os.path.join(MAPS_DIR, "ISOL_*", "*.wav")
CSV_OUT  = "maps_train.csv"
SR      = 44100                # fréquence d’échantillonnage (Hz)
N_FFT   = 2048                 # taille FFT
HOP     = 256 
def wav_to_vec(path: str) -> np.ndarray:
    """WAV mono -> vecteur 88 dim (énergie par touche piano)."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    vec = np.zeros(88)
    for mag, f in zip(S, freqs):
        if f < 20:
            continue
        midi = int(round(69 + 12 * np.log2(f / 440)))
        if 21 <= midi <= 108:
            vec[midi - 21] += mag
    vec /= vec.sum() + 1e-12     # normalisation
    return vec
def midi_from_name(path: str) -> int:
    """Extrait le numéro MIDI du nom de fichier MAPS."""
    name = os.path.basename(path)          # p.ex. ISOL_CH0_Fs4_mf.wav
    note = name.split('_')[-2]             # -> "Fs4"

    pc_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
              'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
              'A#': 10, 'Bb': 10, 'B': 11}

    pitch_class = pc_map[note[:-1]]
    octave = int(note[-1])
    return 12 * (octave + 1) + pitch_class
files = glob.glob(PATTERN)
print(f"{len(files)} fichiers WAV isolés trouvés")

with open(CSV_OUT, "w", newline="") as f:
    wr = csv.writer(f)
    for wav in tqdm.tqdm(files, desc="Extract"):
        wr.writerow([*wav_to_vec(wav), midi_from_name(wav)])

print("✅ Dataset écrit :", CSV_OUT)