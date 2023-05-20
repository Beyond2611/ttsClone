import tkinter as tk
import torch
import torchaudio
import subprocess
import os
import datetime
import IPython
import zoneinfo
from time import time
import argparse

from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tortoise.utils.text import split_and_recombine_text

def normalTTS(text, voice, preset):
    """
    Normal input TTS
    :param text: Text to speak, you can use [] to add expression. EX: "[I'm sad] This sucks" will say "This sucks" in a sad tone
    :param voice: select voice in tortoise/voices , add more voices by creating new directories and adding sample clips
    :param preset: Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py. Highly recommend picking "fast"
    as the quality is good enough and compute in a decent time
    """
    tts = TextToSpeech(models_dir=MODELS_DIR)
    voice_samples, conditioning_latents = load_voice(voice)
    gen = tts.tts_with_preset(text, k=1, voice_samples=voice_samples,
                                         conditioning_latents=conditioning_latents,
                                         preset=preset, cvvp_amount=1)
    print(isinstance(gen,tuple))
    if isinstance(gen, tuple):
        for j, g in enumerate(gen):
            torchaudio.save("Results/Short/Generated_" + f'{voice}_{j}.wav', g.squeeze(0).cpu(), 24000)
    else:
        torchaudio.save("Results/Short/Generated_" + datetime.datetime.now().isoformat() + ".wav",gen.squeeze(0).cpu(), 24000)

    os.makedirs('debug_states', exist_ok=True)
    torch.save(gen, f'debug_states/do_tts_debug_{voice}.pth')

def readTTS(textfile, voice, outpath, preset, regenerate, candidates, seed, produce_debug_state):
    """

    :param textfile: path to textfile
    :param voice: select voice in tortoise/voices.
    :param outpath: output path of the deepfake
    :param preset: see upper
    :param regenerate: Comma-separated list of clip numbers to re-generate, or nothing. Default value should be "None"
    :param candidates: How many output candidates to produce per-voice. Only the first candidate is actually used in the final product, the others can be used manually. Default value
    should be 1
    :param seed: Random seed which can be used to reproduce results. Default value should be 1.
    :param produce_debug_state: Export .pth file for debugging, Default value should be "None"
    """
    tts = TextToSpeech(models_dir=MODELS_DIR)
    if regenerate is not None:
        regenerate = [int(e) for e in regenerate.split(',')]

    # Process text
    with open(textfile, 'r', encoding='utf-8') as f:
        text = ' '.join([l for l in f.readlines()])
    if '|' in text:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        texts = text.split('|')
    else:
        texts = split_and_recombine_text(text)

    seed = int(time()) if seed is None else seed
    voice_outpath = os.path.join(outpath, voice)
    os.makedirs(voice_outpath, exist_ok=True)

    voice_samples, conditioning_latents = load_voice(voice)
    all_parts = []
    for j, text in enumerate(texts):
        if regenerate is not None and j not in regenerate:
            all_parts.append(load_audio(os.path.join(voice_outpath, f'{j}.wav'), 24000))
            continue
        gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                      preset=preset, k=candidates, use_deterministic_seed=seed)
        if candidates == 1:
            gen = gen.squeeze(0).cpu()
            torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), gen, 24000)
        else:
            candidate_dir = os.path.join(voice_outpath, str(j))
            os.makedirs(candidate_dir, exist_ok=True)
            for k, g in enumerate(gen):
                torchaudio.save(os.path.join(candidate_dir, f'{k}.wav'), g.squeeze(0).cpu(), 24000)
            gen = gen[0].squeeze(0).cpu()
        all_parts.append(gen)

    if candidates == 1:
        full_audio = torch.cat(all_parts, dim=-1)
        torchaudio.save(os.path.join(voice_outpath, 'combined.wav'), full_audio, 24000)

    if produce_debug_state:
        os.makedirs('debug_states', exist_ok=True)
        dbg_state = (seed, texts, voice_samples, conditioning_latents)
        torch.save(dbg_state, f'debug_states/read_debug_{voice}.pth')

    # Combine each candidate's audio clips.
    if candidates > 1:
        audio_clips = []
        for candidate in range(candidates):
            for line in range(len(texts)):
                wav_file = os.path.join(voice_outpath, str(line), f"{candidate}.wav")
                audio_clips.append(load_audio(wav_file, 24000))
            audio_clips = torch.cat(audio_clips, dim=-1)
            torchaudio.save(os.path.join(voice_outpath, f"combined_{candidate:02d}.wav"), audio_clips, 24000)
            audio_clips = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default="Short")
    parser.add_argument('--text', default="The quick brown fox")
    parser.add_argument('--voice', default="fauna")
    parser.add_argument('--preset', default="fast")
    args = parser.parse_args()
    #text = "Konfauna, Hololive English Council, Ceres Fauna here"
    #preset = "fast"
    #voice = "fauna"
    outpath = "Results/Long/"
    regenerate = None
    seed = None
    candidates = 1
    produce_debug_state = None
    textfile = "SampleParagraph.txt"
    if(args.type == "Short"):
        normalTTS(args.text, args.voice, args.preset)
    else:
        readTTS(args.text, args.voice, outpath, args.preset, regenerate, candidates, seed, produce_debug_state)



