import json
import logging
import os
from typing import List

from sclib import SoundcloudAPI, Track, Playlist

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("CH")

for lang in ["ar"]:
    base_dir = f"data/tts/vo/{lang}"
    os.makedirs(base_dir, exist_ok=True)
    src_url_file = os.path.join(base_dir, "source_urls.txt")
    tgt_manifest = os.path.join(base_dir, "manifest_dwl.json")
    tgt_agency_map = os.path.join(base_dir, "agency_map.json")
    api = SoundcloudAPI()

    count = 0
    agn_count = 0
    agency_map = {}
    with open(tgt_manifest, encoding="utf-8", mode="w") as tm:
        with open(src_url_file, encoding="utf-8") as s_urls:
            for line in s_urls:
                try:
                    url = line.strip("\n").strip()
                    result = api.resolve(url)
                    tracks: List[Track] = result.tracks if isinstance(result, Playlist) else [result]
                    for track in tracks:
                        agency = str(track.artist).replace(" ", "_").replace(",", "").replace("-", "").replace("/",
                                                                                                               "_").lower()
                        if agency in agency_map:
                            ag_name = agency_map[agency]
                        else:
                            agn_count = agn_count + 1
                            ag_name = f"agn_{lang}_{agn_count}"
                            agency_map[agency] = ag_name

                        tgt_dir = os.path.join(base_dir, ag_name)
                        os.makedirs(tgt_dir, exist_ok=True)
                        f_name = str(track.title).replace(" ", "_").replace(",", "").replace("-", "").replace("/",
                                                                                                              "_").lower()
                        tgt_file = os.path.join(tgt_dir, f"{f_name}.mp3")

                        if not os.path.exists(tgt_file):
                            with open(tgt_file, 'wb+') as file:
                                track.write_mp3_to(file)

                        json.dump({"dwl_af": tgt_file}, tm)
                        tm.write("\n")

                        count = count + 1
                        _log.info(f"Saved: {tgt_file}, lang: {lang}, index: {count}")
                except Exception as e:
                    _log.exception(e)

    with open(tgt_agency_map, encoding="utf-8", mode="w") as tam:
        json.dump(agency_map, tam)
