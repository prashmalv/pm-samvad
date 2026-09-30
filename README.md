# Audio → Indian Sign Language (ISL) Avatar

Speak or upload English audio and get back a video of the sentence in Indian Sign Language, built by joining pre-recorded sign clips.

```
audio ──► faster-whisper (local, CPU) ──► English text
      ──► LangGraph node: AzureChatOpenAI + few-shot prompt ──► ISL gloss  "YOUR NAME WHAT"
      ──► SQLite lookup (gloss → clips/*.mp4), unknown words skipped with a warning
      ──► moviepy (or ffmpeg) concatenation ──► one MP4 returned to the browser
```

## Run it (Python 3.12)

One-time setup: create and activate a virtual environment, then copy `.env.example` to `.env` and fill in your Azure values:

```bash
py -3.12 -m venv venv
```
```bash
venv\Scripts\activate
```
```bash
copy .env.example .env
```

Then the **two commands**:

```bash
pip install -r requirements.txt
```
```bash
python run.py
```

Before starting, `run.py` runs a **pre-flight check** that takes a few seconds, printing one line per dependency:
- **Green ✓:** the dependency is ready.
- **Yellow !:** it works, with a limitation (for example, no alphabet clips means no fingerspelling).
- **Red ✗:** the app would break, for example a missing package, missing Azure settings, no sign database, or port 8000 already in use. Any red item stops the launch.

When everything passes, it prints **"All systems OK — ready for take-off 🚀"** and starts the server. To run only the check:

```bash
python run.py --check
```

Open **http://127.0.0.1:8000**.

On first launch, `run.py` creates `data/isl_clips.db`. The output only contains real signer video: add clips with the importer ([below](#importing-real-sign-videos-islrtc-dictionary-etc)). Any gloss without a clip is skipped. The first transcription downloads the Whisper `base.en` model (~150 MB) from Hugging Face. After that everything except the LLM call runs offline.

> ffmpeg: you don't need a system install. `imageio-ffmpeg` bundles a binary, and moviepy uses it too. If `ffmpeg` is on your PATH, the fallback uses that one.

## Setup on a new machine (from the Git repo)

The repo holds the code, the trained sign model, the 3D signer, the sign database and the avatar motion cache. The **sign videos (`clips/`, about 1.3 GB) are not in Git.** They're shared separately as `SignLang-clips.zip`.

1. Install **Python 3.12**, then clone the repo and create the environment:
   ```bash
   py -3.12 -m venv venv
   ```
   ```bash
   venv\Scripts\activate
   ```
   ```bash
   pip install -r requirements.txt
   ```
2. **Add the sign videos:** unzip `SignLang-clips.zip` into the project folder, so that you get `SignLang\clips\HELLO.mp4` and so on.
3. **Add your Azure settings:** copy `.env.example` to `.env` and fill in your own Azure OpenAI values. `.env` is never committed.
4. **Run it:**
   ```bash
   python run.py
   ```
   The pre-flight check lists anything still missing. Without the clips, Sign → Speech still works, and Speech → Sign shows a warning. The first recording downloads the Whisper model (~150 MB).

Blender, MPFB and the downloaded asset packs (`tools/`, `Downloads/`) are only needed to **rebuild** the 3D signer, not to run the app.

## Environment variables (`.env`)

| Variable | Required | Example / notes |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | yes | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | yes | key from the Azure portal |
| `AZURE_OPENAI_API_VERSION` | yes | e.g. `2024-10-21` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | yes | your **deployment** name (backed by `gpt-4.1` or `gpt-4.1-mini`) |
| `WHISPER_MODEL` | no | `base.en` (default). Use `small.en` for better accuracy, `tiny.en` for speed. |
| `WHISPER_COMPUTE_TYPE` | no | `int8` (default, fastest on CPU) |
| `VIDEO_BACKEND` | no | `auto` (moviepy, then ffmpeg if it fails), `moviepy`, or `ffmpeg` (fastest) |

If any Azure variable is missing, the API returns a 500 error naming the missing variables. Nothing is hardcoded.

## API

| Endpoint | Input | Output |
|---|---|---|
| `GET /topics` | – | Conversation topics (`casual`, `hotel`, `hospital`): label, quick phrases, and each topic word with its clip (`null` = still needs recording) |
| `POST /transcribe` | multipart `audio` file (wav/mp3/m4a/webm/ogg), optional `topic` | `{"text": "..."}` |
| `POST /gloss` | JSON `{"text": "What is your name?", "topic": "casual"}` | `{"text": ..., "gloss": "YOUR NAME WHAT", "tokens": [...], "topic": ...}` |
| `POST /render` | JSON `{"tokens": ["HELLO", "YOUR", "NAME", "WHAT"]}` | `video/mp4`, same headers as below. `segments` gives each sign's start and end time (the UI uses it for word highlighting). |
| `GET /vocabulary` | – | `{"signs": [{"gloss", "clip"}], "fingerspelling": bool, "alphabet": [{"letter", "clip"}]}` |
| `POST /generate-isl-video` | multipart `audio` file | `video/mp4`. The `X-ISL-Info` header holds URL-encoded JSON: `transcript`, `gloss`, `matched`, `missing`, `spelled`, `segments`. If no gloss has a clip, you get a 422 JSON response with the same fields. |

Interactive docs are at http://127.0.0.1:8000/docs.

```bash
curl -X POST http://127.0.0.1:8000/generate-isl-video -F "audio=@hello.wav" -o isl.mp4
```

## Getting sign clips from Hugging Face (recommended)

The demo vocabulary comes from [`bridgeconn/sign-dictionary-isl`](https://huggingface.co/datasets/bridgeconn/sign-dictionary-isl): 3000+ single-sign ISL videos shot against a green screen.

```bash
python scripts/fetch_hf_dataset.py index
```
```bash
python scripts/fetch_hf_dataset.py fetch
```

- `index` runs once and takes about 15–20 minutes. It reads only the tar headers over HTTP range requests and writes `data/hf_index.json`.
- `fetch` downloads just the ~52 demo clips (about 18 MB) instead of the full ~7 GB dataset.

Other useful commands:
- `list <word>` searches the available words.
- `fetch WORD1 WORD2` adds more words.
- `fetch HOME=HOUSE-HOME` saves a dataset word under your own gloss name.

Every clip you fetch automatically becomes part of the LLM's vocabulary.

The dataset comes from an ISL *Bible* dictionary, so some everyday words are missing (PLEASE, NO, SCHOOL, HOSPITAL, TOMORROW, YESTERDAY). Record those yourself and import them with `import_clip.py`.

**Licence:** CC BY-SA 4.0, © Bridge Connectivity Solutions Pvt. Ltd. Credit them in your demo, and share generated videos under the same licence.

## More everyday signs from INCLUDE

The Bible dictionary lacks many everyday words. [INCLUDE](https://zenodo.org/records/4010759) (AI4Bharat, **CC BY 4.0**, Deaf signers) fills part of that gap. It adds about 100 words, for example HOSPITAL, TOMORROW, YESTERDAY, SCHOOL, BATHROOM, RESTAURANT, GOOD-MORNING and TELEVISION:

```bash
python scripts/import_include.py --plan
```
```bash
python scripts/import_include.py
```

- `--plan` lists what would be imported without downloading anything. The second command imports every INCLUDE word the library doesn't have yet. Add words to import only those, or `--replace` to overwrite an existing clip.
- For each word, the importer picks the best of about 15–20 takes using the landmarks from `include_extract.py`. It then streams only that one video from the Zenodo zip, trims it to the sign, crops it to the signer's upper body, and registers it.
- It also adds short aliases (SHOP, ROAD, PHONE, TV, BIG, SMALL, CLOTHES).
- INCLUDE uses several signers against a classroom background, so in **Human** mode the person can change within a sentence. The **3D Avatar** looks the same whichever clip its motion comes from.
- Credit: Sridhar et al., *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*, ACM MM 2020.

## More signs from the ISLRTC ISL Dictionary (non-commercial)

The official [ISLRTC](https://islrtc.nic.in/) dictionary (Govt. of India) is shared on Google Drive. Its A–Z folders hold about 10,800 short word videos. The importer adds every single-word sign the library lacks (about 3,300, for example PLEASE, NO, MY, TEA, COUGH and INJECTION), plus a few useful phrases such as WHAT-TIME and HOW-MANY:

```bash
python scripts/import_islrtc.py index
```
```bash
python scripts/import_islrtc.py --plan
```
```bash
python scripts/import_islrtc.py
```

- `index` lists the Drive folders once and writes `data/islrtc_index.json`. It downloads no videos.
- The main run takes about 2 hours and downloads about 7.7 GB, which is streamed and not kept. It can be resumed: re-run it to continue, for example if Google Drive's download limit stops it.
- Name words to import only those, and add `--replace` to redo one.
- **Cleanup:** the videos place the word label, topic picture and logo differently. MediaPipe finds the signer, then:
  - the clip is trimmed to the signing;
  - everything outside the signer's area (body and wherever the hands move) is painted the video's own background grey, along with graphics left inside it;
  - the signer is never touched.
- Videos longer than 15 s are explanations, not word signs. They're skipped and listed in `data/islrtc_skipped.json`.
- The Drive's other folders (New 2500, MHSL, NCERT) are 12–60 s explainer videos with changing layouts, so they're not imported.
- **Terms** ([ISLRTC FAQ](https://islrtc.nic.in/faq/)): free for research, teaching and ISL technology. It must not be resold or used for profit. Credit *Indian Sign Language Research and Training Centre, DEPwD, Ministry of Social Justice and Empowerment, Govt. of India*. The credit is in the page footer.

## Fingerspelling fallback (names and missing words)

A word with no sign clip is spelled letter by letter with the ISL manual alphabet, e.g. `RAHUL` → R·A·H·U·L. Letters play at 1.5× speed, and the UI shows them in amber with the current letter highlighted. Words are only skipped if they can't be spelled either.

The fallback switches on as soon as the 26 letter clips exist in `clips/alphabet/` (`A.mp4` … `Z.mp4`, and optionally `0.mp4` … `9.mp4`). The Hugging Face dataset doesn't include the alphabet, so record the letters yourself or take them from the ISLRTC alphabet videos. Name each file after its letter, then import the folder:

```bash
python scripts/import_clip.py C:\path\to\letters --alphabet
```

This trims each clip, removes the ISLRTC overlays (add `--keep-overlays` for your own recordings), and converts it to the standard format. Letters are stored outside the gloss DB on purpose, because the dataset already contains the *words* `A` and `I`.

Options in `.env`:
- `FINGERSPELL=0` turns the fallback off.
- `FINGERSPELL_SPEED=1.5` sets the letter playback speed.
- `ALPHABET_DIR` points to a different folder.

## Importing real sign videos (ISLRTC dictionary etc.)

Don't copy raw downloads straight into `clips/`. Run them through the importer:

```bash
python scripts/import_clip.py C:\path\to\Hello.mp4 --gloss HELLO
```
```bash
python scripts/import_clip.py C:\path\to\downloaded_signs
```

For each video, the importer:
1. **Trims** the idle hands-down time at the start and end, using motion detection.
2. **Removes the ISLRTC overlays** (topic picture and label top-left, logo bottom-right), leaving only the signer on the grey background.
3. **Normalizes** to 1280×720, 25 fps, H.264, no audio.
4. **Saves** the result as `clips/<GLOSS>.mp4` and **registers** it in the DB, which also adds it to the LLM's vocabulary.

In folder mode, each gloss comes from the filename: `Thank_You.mp4` → `THANK-YOU`. Use `--keep-overlays` for your own recordings and `--no-trim` to keep the full length.

**Word signs vs. explanation videos:** ISLRTC publishes both. A *word* video shows one sign (about 1–4 s) and is what sentence stitching needs. An *explanation* video, like `Adipose_Cells.mp4` (22 s), describes a concept in ISL. It imports fine as one gloss, but it's too long to use as a word inside a sentence. The importer warns you when a clip is longer than 6 s.

## Adding your own clips

After `fetch --all`, `clips/` holds about 2850 signs. Words the dataset lacks (PLEASE, NO, SCHOOL, HOSPITAL, TOMORROW, YESTERDAY …) are fingerspelled once the alphabet is installed. For better results, record real signs for the everyday ones and import them with `import_clip.py`. Each clip is saved as `clips/<GLOSS>.mp4`. For pipeline testing without real clips, `python scripts/seed_db.py --placeholders` creates title-card stand-ins.

**Where to get the correct signs.** Use the official **ISLRTC Indian Sign Language Dictionary** (Indian Sign Language Research and Training Centre, Govt. of India). It has a reference video for each word. Learn the sign from it and record it yourself, or check that dataset's license before reusing its footage. Research datasets such as **INCLUDE** (AI4Bharat) and **ISL-CSLTR** also contain word-level ISL clips; check each one's license too. Regional ISL variants exist, so ideally have a Deaf signer or ISL interpreter check the clips before a public demo.

**Recording spec (makes the stitched video look like one continuous signer):**
- Same signer, same clothing, plain background, same camera position and lighting for every clip.
- Landscape 16:9, framed from the waist up with room above the head for signs near the face. Any resolution works: clips are letterboxed to 1280×720 at 25 fps when stitched.
- **Start and end every clip in the same neutral pose** (hands resting at waist). This is the most important rule for smooth joins.
- Trim to the sign itself, roughly 0.8–2 s. Audio is ignored.
- Format: MP4 (H.264). If a phone records HEVC/MOV, convert it:
  `ffmpeg -i IMG_0001.MOV -c:v libx264 -an HELLO.mp4`

**Adding new words:** `python scripts/import_clip.py MyClip.mp4 --gloss WORD`. The importer registers the word in the DB, and it automatically joins the LLM's preferred vocabulary.

## Human or 3D Avatar signer (Phase 1)

The Speech → Sign page has a **Human | 3D Avatar** switch above the video:

- **Human:** the real signer clips, stitched into an MP4 (as described above).
- **3D Avatar:** a figure drawn in the browser with three.js and animated from the *same* clips. MediaPipe extracts each clip's 3D body and hand motion, which is mapped onto a fixed skeleton. The result is one consistent figure, signs that **blend into each other** (the rest pose between signs is removed), and a view you can rotate by dragging. **Download** records the avatar to a `.webm` file.

It's completely free. three.js (MIT) and MediaPipe (Apache 2.0) are open source, and the avatar is built from simple 3D shapes in code ([static/avatar.js](static/avatar.js)), so there are no character models or licences involved.

```bash
python scripts/build_avatar_motion.py
```
pre-computes the demo signs (about 5 s per sign on CPU; `--all` does every clip). Anything not pre-computed is extracted the first time it's requested, then cached in `data/avatar_motion/`.

**Limits:** the motion is estimated from ordinary 2D video, so depth (how far forward the hands are) is approximate, and fast finger movements can blur. Treat it as a demo and have an ISL signer check it before relying on it. The Human mode remains the reference.

### Realistic human signer (MPFB2)

When `static/models/signer.glb` exists, 3D Avatar mode shows a realistic person instead of the simple figure. It's a young Indian woman with a ponytail and a dark top, and the simple figure remains the fallback. She was made with **MPFB2** (the MakeHuman add-on for Blender), and the character and its assets are **CC0**.

[static/avatar-human.js](static/avatar-human.js) turns the same joint motion into bone rotations:
- **Arms:** each bone is turned to point where the video's joints point.
- **Fingers** ([static/avatar-hands.js](static/avatar-hands.js)) move the way real fingers can:
  - The knuckle and the two finger joints bend only towards the palm, within human limits.
  - Sideways spread is small, and the end joint follows the middle joint.
  - Each angle is median-filtered over 0.28 s, and the ring and pinky partly follow their neighbours.
  - When fingers hide each other, as in a fist, the tracked angles come out impossible. The curl is then taken from how close each fingertip is to the palm.
- **Hands:** full palm orientation.
- **Head:** turned to face where the video's face points.
- **Blinking:** natural, using the ARKit face shapes.

**Rebuilding the character.** Blender and MPFB run headless, so no clicking is needed:
1. Unzip Blender 5.2 into `tools/`. Create the empty folder `tools/blender-5.2.2-windows-x64/portable`, so Blender keeps its settings there.
2. Install MPFB2 and its asset packs from `Downloads/avatar/`. The packs are makehuman_system_assets, skins01/02, hair01, shirts01, pants01 and faceunits01.
   ```bash
   tools/blender-5.2.2-windows-x64/blender.exe -c extension install-file -r user_default --enable Downloads/avatar/add-on-mpfb-v2.0.17.zip
   ```
   ```bash
   tools/blender-5.2.2-windows-x64/blender.exe --background --python scripts/avatar/install_mpfb_assets.py
   ```
3. Build the character:
   ```bash
   tools/blender-5.2.2-windows-x64/blender.exe --background --python scripts/avatar/build_signer.py
   ```
   This takes about 20 seconds and writes `static/models/signer.glb` (about 5 MB) and the editable `data/avatar/signer.blend`.

To change the look (gender, body, skin, hair, clothes), edit the settings at the top of [scripts/avatar/build_signer.py](scripts/avatar/build_signer.py) and rebuild. Colour tweaks such as the dark top live in `LOOK` in `avatar-human.js`.

To check the character, open `/dev/avatar-test.html?words=HELLO,NAME` and call `show(<seconds>)` in the browser console.

## Phase 2: Sign → Speech (http://127.0.0.1:8000/sign.html)

Sign one word at a time in front of your webcam, or upload a video. The app recognises each sign, turns the sequence into an English sentence, and reads it aloud.

```
webcam / video ──► MediaPipe hand + pose tracking (in the browser, video never uploaded)
               ──► split into signs: hands raised … hands lowered
               ──► POST /recognize: landmark points → Transformer (ONNX) → top-5 signs + confidence
               ──► POST /to-english: gloss words → GPT-4.1 (LangGraph node) → "I want water."
               ──► browser text-to-speech (free, offline voice)
```

- **Model:** a small Transformer over 32 frames of 55 body and hand points. It's trained on **INCLUDE** (AI4Bharat, CC BY 4.0): 263 everyday ISL word signs, about 4,300 videos by Deaf signers. It runs on CPU with `onnxruntime`, so the app doesn't need PyTorch.
- **Consistent tracking:** the browser and the training data both use MediaPipe **1.0.1** with the same `.task` models (`static/models/`), so live input matches what the model saw.
- **Resting hands:** INCLUDE films the full body, but webcams usually don't see resting hands. Hands below the rest line (`REST_Y` in [app/sign_features.py](app/sign_features.py)) are ignored in both cases.
- **Picking another guess:** click a recognised sign to choose one of the other top-5 guesses or remove it, before making the sentence.

**Training it yourself** (one-time; the result is saved in `models/sign/`):

```bash
pip install -r requirements-train.txt
```
```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```
```bash
python scripts/include_extract.py
```
```bash
python scripts/train_sign_model.py
```

`include_extract.py` never stores the 56.8 GB of videos. It reads the Zenodo zips with HTTP range requests, streams each video through MediaPipe on 8 parallel workers, and keeps only the landmarks (about 30 MB). It's resumable, and Zenodo's download speed makes a full run take about 4 hours. The trainer uses INCLUDE's official train/val/test split and reports held-out test accuracy, which the Sign → Speech page shows.

| Endpoint | Send | Get back |
|---|---|---|
| `POST /recognize` | `{"frames": [[110 numbers or null] × T]}`, one sign | `{"candidates": [{"gloss", "confidence"}]}` |
| `POST /to-english` | `{"glosses": ["I", "WATER", "WANT"]}` | `{"english": "I want water."}` |
| `GET /sign-model` | – | known signs + test accuracy |

## Conversation topics (Casual · Hotel · Hospital)

On first open the Speech → Sign page asks what the conversation is about. The topic is remembered, and you can change it from the top bar. Topics are defined in [app/topics.py](app/topics.py), and each one sets:

- **Setting for the LLM:** ambiguous words are read the way the setting means them. For example, "check out" becomes LEAVE at a hotel, and medical meaning is kept exact at a hospital.
- **Topic signs:** always offered to the LLM as preferred vocabulary, plus a few topic-specific examples.
- **Whisper hint:** the topic words are passed as `initial_prompt`, so words like "fever" and "towel" are heard correctly.
- **UI:** each topic has its own quick phrases, and its signs appear first in the Sign library. Signs without a clip yet are greyed out there.

To add a word to a topic, add it to the topic's `words`. If it has no clip yet, record one and import it with `import_clip.py`.

## How the gloss prompt works

[app/gloss_graph.py](app/gloss_graph.py) is a two-node LangGraph graph: `translate` (AzureChatOpenAI, `temperature=0`) → `normalize` (deterministic clean-up).

- **Rules:** drop articles and copulas; order is time first, then topic, then comment, with SOV inside each clause; question words go at the end; NOT goes after the verb; verbs stay in base form, with past shown by a time word or FINISH; uppercase output, hyphenated multi-word signs.
- **11 few-shot examples**, each chosen to show one rule, e.g. `What is your name? → YOUR NAME WHAT`, `I don't understand. → I UNDERSTAND NOT`, `I am so glad to meet you! → I YOU MEET HAPPY`.
- **Preferred vocabulary:** glosses with clips are sent in the system prompt. Once there are more than 300, only those sharing a word stem with the sentence are sent, plus about 50 core everyday signs, which keeps the prompt small. The model maps synonyms onto clips that exist ("glad" → HAPPY, "house" → HOME). If nothing matches, it keeps the plain word instead of forcing a wrong sign.
- **Normalizer:** removes stray punctuation and prefixes like `GLOSS:`, plus any leftover articles or copulas.
- Words with no clip, like names (`MY NAME RAHUL`), are **fingerspelled** when the alphabet is installed. Otherwise they are **skipped with a warning** and shown crossed out in the UI.

## Project layout

```
app/
  config.py        env vars, paths, output format
  transcriber.py   faster-whisper (CPU, int8, VAD)
  gloss_graph.py   LangGraph + AzureChatOpenAI + few-shot prompt
  db.py            SQLite gloss -> clip lookup (warn on misses)
  fingerspell.py   letter-by-letter fallback from clips/alphabet/
  video.py         moviepy stitching with ffmpeg subprocess fallback
  pipeline.py      end-to-end orchestration
  main.py          FastAPI endpoints + serves static/
static/            single-page UI: index.html, styles.css, app.js (no framework)
scripts/seed_db.py creates data/isl_clips.db (+ --placeholders test clips)
scripts/import_clip.py  trims, cleans and registers real sign videos (--alphabet for letters)
scripts/fetch_hf_dataset.py  index + download clips from the Hugging Face dataset
clips/             ISL clips (GLOSS.mp4); clips/alphabet/ holds A.mp4..Z.mp4
data/              SQLite DB
outputs/           stitched videos (last 30 kept)
run.py             seeds DB if needed, starts uvicorn
```

## Troubleshooting

- **Recording button does nothing:** browsers only allow the microphone on `localhost`/`127.0.0.1` or HTTPS. Use the URL above.
- **Every gloss is skipped:** run `python scripts/seed_db.py` and check which files it reports as missing, and whether the filenames match exactly.
- **Stitching is slow:** set `VIDEO_BACKEND=ffmpeg` in `.env`. It is several times faster than moviepy.
- **401/404 from Azure:** `AZURE_OPENAI_DEPLOYMENT_NAME` must be the deployment name, not the model name, and the endpoint must be the resource root URL.
