import pysubs2 as pysubs2

src_srt = "/Users/Shahin.Konadath/Downloads/voice_overs/new_en.srt"

subs = pysubs2.load(src_srt, encoding="utf-8")
for s_index, sub in enumerate(subs):
    print(sub.text)

