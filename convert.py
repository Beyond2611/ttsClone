import os
from pydub import AudioSegment
import torchaudio
import argparse

def convertMP3toWAV(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type = str, help = 'MP3 source', default = os.curdir)
    parser.add_argument('--dst', type = str, help = 'WAV output destination', default = os.curdir)
    args = parser.parse_args()
    convertMP3toWAV(args.src, args.dst)