# Jarvis

A local, voice-driven AI companion for Windows 11. Say "Jarvis" once and he stays listening until you say "Jarvis stop".
You speak French or English; he always answers in English, with a calm, slightly witty Iron Man vibe.

Everything runs on your machine: Ollama (LLM), faster-whisper on CUDA (speech-to-text), Piper (text-to-speech).
Only web search touches the internet.

## Setup

Requirements: Windows 11, Python 3.10-3.12, an NVIDIA GPU with a recent driver, [Ollama](https://ollama.com) installed.

1. Double-click **`start.bat`** (or run it from a terminal). The first run:
   - creates `.venv` and installs `requirements.txt` (including the CUDA 12 cuBLAS/cuDNN wheels Whisper needs),
   - downloads the Whisper model (~1.5 GB for `medium`) into `models/whisper/`,
   - downloads the Piper voices you use into `models/piper/`,
   - pulls the Ollama model (`qwen2.5:14b`, ~9 GB) if it is missing.
2. Wait for `== SLEEPING - say 'Jarvis' ... ==`, then talk, or type in the console.

Manual run: `.venv\Scripts\activate`, then `python main.py` (`python main.py --text` disables the microphone).

If Ollama isn't running, Jarvis tries to start `ollama serve`; if Ollama isn't installed he says so with the download link
(https://ollama.com/download) and exits. Missing models are pulled automatically (`llm.auto_pull`).
If no microphone is found he says so and keeps working with typed input.

**No NVIDIA GPU / CUDA?** At startup Jarvis detects the hardware. Without a usable CUDA GPU he switches (in memory, `config.yaml` is
untouched) to `stt.device: cpu`, `compute_type: int8` and the `small` Whisper model, and logs it. With under 12 GB of VRAM he logs a
suggestion to use the 7B models (`qwen2.5:7b`, `qwen2.5-coder:7b`, see `config.yaml`). `python hardware.py` shows what was detected.

## Moving to another PC

Copy the whole `Jarvis` folder **except** `.venv\`, `workspace\.venv\` and `data\` (virtual environments are tied to the PC they were
built on; `data\` holds this PC's memories and settings, copy it only if you want to keep them). Then install Python 3.10-3.12 and
Ollama on the new PC and run **`start.bat`**. It finds a compatible Python, deletes any broken or copied `.venv` /
`workspace\.venv`, rebuilds the environment, runs `pip install -r requirements.txt` every time (instant when already satisfied) and
launches Jarvis with the venv's own Python. Copying `models\` is optional; missing models are downloaded again.

## Using him

| Say / type | Effect |
|---|---|
| `Jarvis` | wakes him; he stays active |
| `Jarvis, what's the weather in Lyon?` | wake word and request in one go |
| `Jarvis stop` | back to sleep, wake word needed again (`arrête`, `that's all`, `go to sleep` also work) |
| (5 minutes of silence) | he says "Going quiet, sir." and sleeps (`wake_word.inactivity_timeout_s`; 0 = never). The timer is paused during a yes/no question or a running request. |

**Waking rule.** While asleep, "Jarvis" only counts at the *start* of what you say ("Jarvis, ...", "Hey Jarvis, ...") or in a very short
utterance (3 words or fewer, like "wake up Jarvis"). "I told Jarvis to turn the lights off" does not wake him.

**Console commands:** `/help`, `/mic` (input devices, which one is in use, 2 s level test; diagnoses a silent mic), `/status` (loaded
models and GPU %, voice, mute, awake/asleep, workspace), `/reset`, `/quit`.
| `change voice` / `list voices` / `use the cyn voice` | voice control |
| `mute` / `unmute` | mute = text-only replies (saved across restarts) |
| `remember that my dog is called Max` | stored in long-term memory |
| `forget my dog` / `forget everything` (asks to confirm) | removes memories |
| `what do you know about me?` | lists what he remembers |
| `shut down` | exits |

You can also just talk to him: facts you mention ("I work night shifts") are auto-extracted into memory in the background and
injected into later prompts when relevant. Console-only commands: `/help`, `/reset` (clear chat history), `/quit`.

Speak over him and he stops talking (barge-in). With speakers he can hear himself, so this needs a louder voice
than normal; headphones make it reliable. Tune or disable it under `audio:` in `config.yaml`.

## Coding

Jarvis can write, edit, read, list and run code in a private **workspace** folder (`./workspace`, set by `code.workspace`).
Coding requests ("write a script that...", "code me...", "fix this code", "make a program...") are routed automatically to the
coder model (`llm.coder_model`, default `qwen2.5-coder:14b`); everything else uses the chat model. Short follow-ups
("run it", "yes", "add a score counter") within three minutes stay on the coder. Code is never read aloud: you get a short spoken
summary and the code appears in the console.

| Say / type | Effect |
|---|---|
| `write a script called fib.py that prints the first 20 Fibonacci numbers and run it` | writes, shows, asks, runs, reports |
| `open workspace` | opens the folder in Explorer |
| `list files` | lists the workspace |
| `show snake.py` | prints the code in the console |
| `run snake.py` | runs it (code shown, confirmation first); on failure he reads the error and fixes it |

**Safety**
- All file access is confined to the workspace (`..`, other drives and absolute paths elsewhere are refused). Only text/source
  extensions can be written, never `.exe`/`.bat`/etc.
- **Running a script, deleting a file and installing a pip package always ask first.** Say "yes"/"no" or type `y`/`n`. Silence for
  90 s, or anything unclear, counts as no. The full code is printed before every run. Ambient speech cannot approve while he is asleep.
- Scripts run in their **own virtual environment, `workspace/.venv`** (created on first use), never in Jarvis's `.venv`; `pip_install`
  installs there too. They run with the workspace as the working directory, a 60 s timeout and no keyboard input. The file tools
  refuse to touch, and `list_files` hides, `.venv`.
- **Risk scan before every approval.** The code is scanned and a `WARNING` lists risky things with line numbers: file deletes
  (`os.remove`, `shutil.rmtree`, `Path.unlink`), `open()` on absolute / `..` / computed paths, `subprocess` / `os.system`,
  network modules, registry access, `eval`/`exec`. If anything is found, the spoken question becomes "This script does something
  risky, check the console. Run it?". The scan is best-effort static analysis, not a sandbox: it cannot see code built at run time or
  inside other modules. Scripts still run as you, so read the code before saying yes.
- A failed run is fixed and retried at most 3 times per request.

Note on VRAM: the two 14B models do not fit in 12 GB together with Whisper, so Ollama swaps them; the first reply after switching
between chat and coding takes a few extra seconds. Set `llm.coder_model: ""` to use one model for everything.
At startup (after Whisper and the chat model are loaded) Jarvis asks Ollama's `/api/ps` where the model lives and logs the GPU/CPU
split; if it is not 100% on the GPU he warns you and suggests the 7B models, which are in `config.yaml` as commented alternatives
(`qwen2.5:7b`, `qwen2.5-coder:7b`). `/status` shows the same at any time.

## Voices

| key | voice |
|---|---|
| `jarvis` | clean British male (Piper `en_GB-alan-medium`) |
| `ryan` | clean American male (`en_US-ryan-high`) |
| `amy` | clean American female (`en_US-amy-medium`) |
| `cyn` | dark robotic Cyn-style: Amy + pitch down, ring modulation, bitcrush, overdrive, reverb |

Voices are defined in `config.yaml` under `tts.voices`. Add any [Piper voice](https://huggingface.co/rhasspy/piper-voices)
by its name (it is auto-downloaded), and shape it with an effect chain (`pitch`, `ring_mod`, `bitcrush`, `overdrive`,
`lowpass`, `highpass`, `reverb`; see `voices/effects.py`). Tweak `cyn` freely, for example raise `ring_mod.mix` for more metal.

## Configuration

`config.yaml` holds everything: LLM model and persona, Whisper model (`medium` or `large-v3-turbo`), VAD sensitivity,
voices, memory, web search. Voice choice and mute state are saved to `data/settings.json`; memories live in `data/memory.db`.

Tips:
- Choose the microphone with `audio.input_device`: `""` = Windows default, or part of its name (`"SoloCast"`), or an index. Run `/mic`
  to see the list. Nothing measures the mic at startup; `/mic` is the diagnostic.
- Voice commands that Whisper mishears: any transcript containing "voice"/"voix" that matched no command is logged as
  `Voice-like phrase not matched: '...'`; add the mishearing to that voice's `aliases` in `config.yaml`.
- Room too noisy / he reacts to noise: raise `audio.vad_min_rms` (e.g. 0.015).
- He cuts you off mid-sentence: raise `audio.silence_ms`.
- A lighter model for fact extraction: `ollama pull qwen2.5:3b`, then set `memory.extract_model: qwen2.5:3b`.
- Wake word misheard: add the spelling you see in the console (`Heard [en]: ...`) to `wake_word.aliases`.

## Project layout

```
main.py            orchestration: threads, wake/active state, event loop
commands.py        EN/FR command router (voice, mute, memory, stop) that never hits the LLM
stt.py             microphone capture + device selection + adaptive VAD + faster-whisper (FR/EN auto-detect)
wakeword.py        fuzzy wake-word (strict while sleeping) and stop-phrase detection
tts.py             Piper TTS, voice management, interruptible two-stage playback
voices/effects.py  numpy/scipy effects (pitch, ring mod, bitcrush, overdrive, filters, reverb)
llm.py             Ollama streaming chat, tool calling, history trimming, memory injection
memory.py          SQLite long-term memory + LLM fact extraction
tools/             tool registry + web_search.py (DuckDuckGo) + code.py (sandboxed workspace, run, pip)
common.py          config, logging, settings
hardware.py        startup GPU/CUDA/VRAM detection and CPU fallback
start.bat          portable launcher: finds Python, repairs .venv, installs requirements, runs main.py
```

## Testing modules on their own

```
python tts.py [--save|--no-play]   speaks a line in every voice (--no-play: silent, writes WAVs to data/tts_test/)
python stt.py [--mic]       TTS->STT round trip; with --mic, live transcription from your microphone
python llm.py "question"    one LLM turn, including tool calls
python -m tools.web_search "query"
python wakeword.py | python memory.py | python commands.py | python -m voices.effects
python -m tools.code        sandbox, venv isolation, risk scan, confirmation and retry-budget tests (no LLM needed)
python test_jarvis.py       config defaults, /mic + /status, VRAM report, inactivity timer, confirmation timer (no LLM, silent)
python test_voice_switching.py   voice commands through Jarvis.handle (silent: writes WAVs, no playback)
python test_voice_pipeline.py    TTS WAV -> VAD -> Whisper -> wake word -> command router -> set_voice (silent, no mic)
python test_coding.py       end-to-end coding with the real models: fib.py, a declined run, and a bug-fix loop
```
None of the tests plays audio or uses the microphone.

## Troubleshooting

- **`Could not load library cudnn...` / Whisper falls back to CPU**: re-run `pip install -r requirements.txt` inside the venv;
  Jarvis adds the pip-installed NVIDIA DLL folders to the DLL search path automatically.
- **No sound**: set `tts.output_device` in `config.yaml` (list devices with `python -m sounddevice`).
- **He doesn't hear me / wrong microphone**: type `/mic`. It lists the input devices, marks the one in use and measures 2 s of
  level; if the mic is silent it lists the likely causes (hardware mute, Windows microphone privacy for desktop apps, wrong device).
  Then set `audio.input_device` to part of the right name.
