import os
import argparse
from math import floor

from pydub import AudioSegment

def AudioSlicer(src, dst):

    audio = AudioSegment.from_wav(src)
    last = 0
    iter = 1
    Segment = audio[:1000]
    for it in range(10, floor(audio.duration_seconds), 10):
        if(last == 0):
            Segment = audio[:10000]
            last = it
            Segment.export(dst + f'{iter}' + ".wav", format="wav")
        else:
            Segment = audio[last * 1000:last * 1000 + 10000]
            Segment.export(dst + f'{iter}' + ".wav", format="wav")
            last = it
        iter = iter + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='WAV source', default=os.curdir)
    parser.add_argument('--dst', type=str, help='WAV slices output destination', default=os.curdir)
    args = parser.parse_args()
    AudioSlicer(args.src, args.dst)