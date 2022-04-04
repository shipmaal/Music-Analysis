from os.path import dirname, join, isfile

import wandb
from numToWord import numToWord

from dataclasses import dataclass

import torch
import torchaudio

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()


def make_transcript(path):
    f = open(path, 'r')
    lyriclist = f.readlines()[2 if f.readline(1) == '\n' else 1:]
    f.close()

    for char in (lyriclist[-1]):
        if char.isdigit():
            lyriclist[-1] = lyriclist[-1].split(char)[0]
            break
    temp = ""

    for i in lyriclist:
        temp += i

    lyriclist = temp

    for i in ['\n', ' ', '-', '/']:
        lyriclist = lyriclist.replace(i, '|')

    for i in ['(', ')', ',', '.', '?', '!', '’']:
        lyriclist = lyriclist.replace(i, '')

    lyriclist = lyriclist.replace('$', 's')
    lyriclist = lyriclist.replace('í', 'i')
    lyriclist = lyriclist.replace('ü', 'u')

    temp = lyriclist.split('|')
    for i in temp:
        temp_str = ""
        if any(char.isdigit() for char in i):
            for j in i:
                if j.isdigit():
                    temp_str += j
            temp[temp.index(i)] = numToWord(int(temp_str))
    lyriclist = ""
    for i in temp:
        lyriclist += f"{i}|"
    return lyriclist


print(make_transcript(r'C:\Users\atshi\PycharmProjects\MusicTest\Projects\Who Dat Boy\Lyrics\Who Dat Boy.txt'))


def grab_times(audiofile: str, txtfile: str):
    with torch.inference_mode():
        waveform, sample_rate = torchaudio.load(audiofile)
    emission_path = join(dirname(dirname(audiofile)), 'emission.pt')
    if isfile(emission_path):
        emission = torch.load(join(dirname(dirname(audiofile)), 'emission.pt'))
    else:
        print(waveform.size(1) / sample_rate)
        print(waveform.size())
        # code to allow this to run on shitty computers
        frames = sample_rate * 15
        for i in range(0, round(waveform.size(1) / frames)):
            waveform_temp = waveform[:,
                            (frames * i):(frames * (i + 1)) if (frames * (i + 1)) < waveform.size(1) else None]
            print(waveform_temp.size())
            emissions_temp, _ = model(waveform_temp.to(device))
            emissions_temp = torch.log_softmax(emissions_temp, dim=-1)
            print(i)
            if i == 0:
                emission = emissions_temp[0].cpu().detach()
            else:
                emission = torch.cat([emission, emissions_temp[0].cpu().detach()], dim=0)
        torch.save(emission, join(dirname(dirname(audiofile)), 'emission.pt'))
        emission = torch.load(join(dirname(dirname(audiofile)), 'emission.pt'))

    print(emission.size())

    transcript = make_transcript(txtfile)
    transcript = transcript.upper()
    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]

    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.full((num_frame + 1, num_tokens + 1), -float('inf'))
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    trellis = get_trellis(emission, tokens)

    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float

    def backtrack(trellis, emission, tokens, blank_id=0):
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            path.append(Point(j - 1, t - 1, prob))

            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError('Failed to align')
        return path[::-1]

    path = backtrack(trellis, emission, tokens)

    ratio = waveform.size(1) / (trellis.size(0) - 1)

    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            x0 = int(ratio * self.start)
            x1 = int(ratio * self.end)

            return f"{self.label}\t({self.score:4.2f}), [{x0 / sample_rate:.3f} {x1 / sample_rate:.3f})"

        @property
        def length(self):
            return self.end - self.start

        @property
        def time(self):
            return [(ratio*self.start)/sample_rate, (ratio*self.end)/sample_rate]

    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2 - 1].time_index + 1, score))
            i1 = i2
        return segments

    segments = merge_repeats(path)

    def merge_words(segments, separator='|'):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = ''.join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words

    return merge_words(segments)


for word in (grab_times(r"C:\Users\atshi\PycharmProjects\MusicTest\Projects\Who Dat Boy\Audio\Who Dat Boy.ogg",
                        r"C:\Users\atshi\PycharmProjects\MusicTest\Projects\Who Dat Boy\Lyrics\Who Dat Boy.txt")):
    print(word, word.time)
