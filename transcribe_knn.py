import argparse
import librosa
import numpy as np
import joblib
import mido

SR = 44100
N_FFT = 2048
HOP = 256
HOP_SEC = HOP / SR

MODEL = joblib.load("knn_model.pkl")


def wav_to_mat(path: str) -> np.ndarray:
    """WAV mono -> matrice (frames × 88)."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    V = np.zeros((S.shape[1], 88), dtype=np.float32)
    for k, f in enumerate(freqs):
        if f <= 0: 
        continue
        midi = int(round(69 + 12 * np.log2(f / 440)))
        if 21 <= midi <= 108:
            V[:, midi - 21] += S[k]
    if V.sum(axis=1, keepdims=True) != 0:
    V /= V.sum(axis=1, keepdims=True) 
    return V


def frames_to_events(pred) -> list[tuple[int, float, float]]:
    """Regroupe les prédictions frame-par-frame en notes continues."""
    events = []
    prev = pred[0]
    start = 0
    for i, n in enumerate(pred[1:], 1):
        if n != prev:
            if prev:
                events.append((prev, start * HOP_SEC, (i - start) * HOP_SEC))
            prev = n
            start = i
    if prev:
        events.append((prev, start * HOP_SEC, (len(pred) - start) * HOP_SEC))
    return events


def write_midi(events, out_path, tempo=500_000, ppq=480):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    for pitch, st, dur in events:
        tick_on = int(st / (60 / 120) * ppq)      # tempo 120 bpm
        tick_off = int((st + dur) / (60 / 120) * ppq)
        track.append(mido.Message('note_on', note=pitch, velocity=90, time=tick_on))
        track.append(mido.Message('note_off', note=pitch, velocity=0, time=tick_off))
    mid.save(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="WAV mono 44 kHz")
    ap.add_argument("--midi",  required=True, help="chemin de sortie .mid")
    args = ap.parse_args()

    X      = wav_to_mat(args.audio)
    notes  = MODEL.predict(X)
    events = frames_to_events(notes)
    write_midi(events, args.midi)

    print(f" MIDI généré ({len(events)} notes) → {args.midi}")
