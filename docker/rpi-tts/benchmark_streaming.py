"""Streaming TTS benchmark for RPi4 — măsoară TTFA, RTF, gap-uri, peak RAM."""
import time
import sys
import re
import numpy as np
from pathlib import Path

# Model
sys.path.insert(0, "/app")
from supertonic.loader import load_model, load_voice_style_from_name

MODEL_DIR = Path("/root/.cache/supertonic3")
VOICE = "F1"
LANG = "ro"
STEPS = 5
SPEED = 1.5

# ========== CHUNK LOGIC ==========
def split_sentences(text: str):
    """Split text at sentence boundaries .?!;: — return list of sentences."""
    # Split at sentence-ending punctuation, keeping the punctuation
    parts = re.split(r"(?<=[.?!;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text_stream(text: str, first_max: int = 80, next_max: int = 220):
    """
    Chunk text for streaming:
    - First chunk: up to `first_max` chars (short, for fast TTFA)
    - Subsequent chunks: up to `next_max` chars
    - Split at sentence boundaries when possible
    """
    sentences = split_sentences(text)
    if not sentences:
        return [text]
    
    chunks = []
    current = ""
    max_for_phase = first_max  # first chunk uses first_max
    
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_for_phase:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
                max_for_phase = next_max  # switch to next_max after first chunk
            current = sent
    
    if current:
        chunks.append(current)
    
    # If only 1 chunk, use it
    if len(chunks) == 1:
        return chunks
    
    # Ensure first chunk is at least some text (not tiny)
    if len(chunks[0]) < 10 and len(chunks) > 1:
        # Merge with next chunk
        chunks[0] = chunks[0] + " " + chunks[1]
        chunks.pop(1)
    
    return chunks


# ========== BENCHMARK ==========
print(f"{'='*70}")
print(f"STREAMING TTS BENCHMARK — RPi4")
print(f"{'='*70}")
print(f"Model: FINAL (ve INT8 + rest FP32)")
print(f"Config: steps={STEPS}, speed={SPEED}, voice={VOICE}, lang={LANG}")
print(f"")

# Load model once
print("Loading model...", flush=True)
t0 = time.time()
tts = load_model(MODEL_DIR, auto_download=False)
style = load_voice_style_from_name(MODEL_DIR, VOICE)
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

LONG_TEXT = (
    "Bună ziua! Acesta este un test pentru sistemul text-to-speech în limba română. "
    "Funcționează corect atât pe procesor cât și în container Docker. "
    "Am testat mai multe configurații și am ales cea mai optimă variantă. "
    "Sistemul poate citi texte lungi fără probleme. "
    "Calitatea sunetului este foarte bună pentru un model de 99 de milioane de parametri. "
    "Acesta rulează integral pe un Raspberry Pi 4 cu patru nuclee ARM Cortex-A72. "
    "În ciuda limitărilor hardware, performanța este acceptabilă pentru un asistent vocal."
)

SHORT_TEXT = "Bună ziua! Acesta este un test pentru sistemul text-to-speech în limba română."


def run_benchmark(text: str, label: str, n_runs: int = 5):
    """Run benchmark for a given text, return all metrics."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {label}")
    print(f"Text: {len(text)} chars, ~{len(text.split())} words")
    print(f"{'='*70}")
    
    chunks = chunk_text_stream(text)
    print(f"Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {len(c):3d} chars — \"{c[:60]}...\"")
    
    results = []
    
    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...", flush=True)
        
        # Memory before
        import psutil
        mem_before = psutil.Process().memory_info().rss / 1e6
        
        chunk_times = []
        t_global_start = time.time()
        
        for i, chunk in enumerate(chunks):
            t0 = time.time()
            wav, dur = tts([chunk], style, total_step=STEPS, speed=SPEED, lang=LANG)
            t1 = time.time()
            
            gen_time = t1 - t0
            audio_dur = float(dur[0])
            latency_from_start = t1 - t_global_start
            
            chunk_times.append({
                "chunk": i,
                "chars": len(chunk),
                "gen_time": gen_time,
                "audio_dur": audio_dur,
                "rtf": gen_time / audio_dur if audio_dur > 0 else 0,
                "latency": latency_from_start,
            })
            
            print(f"    Chunk {i}: {gen_time:.2f}s gen, {audio_dur:.2f}s audio, "
                  f"RTF={gen_time/audio_dur:.3f}, latency={latency_from_start:.2f}s", flush=True)
            
            # Simulate playback delay (non-blocking — just track time)
            # In real scenario, playback would be parallel
        
        t_total = time.time() - t_global_start
        total_audio = sum(c["audio_dur"] for c in chunk_times)
        
        # Calculate gaps: time between chunk i done and chunk i+1 ready
        gaps = []
        for i in range(len(chunk_times) - 1):
            chunk_end = chunk_times[i]["latency"]
            next_chunk_ready = chunk_times[i+1]["latency"]
            playback_end = chunk_times[i]["latency"]  # chunk i finished generating at this time
            # If sequential (no parallel), gap = next_chunk_ready - (chunk_end + chunk_audio_dur)
            # But if next chunk starts gen immediately after chunk i, and playback runs in parallel:
            gap = chunk_times[i+1]["latency"] - (chunk_times[i]["latency"])
            gap_vs_playback = (chunk_times[i+1]["latency"]) - (chunk_times[i]["latency"] - chunk_times[i]["gen_time"] + chunk_times[i]["audio_dur"])
            gaps.append({
                "between": f"{i}→{i+1}",
                "gap_sequential": round(gap - chunk_times[i]["audio_dur"], 3),
                "gap_parallel_best": round(gap_vs_playback, 3),
            })
        
        mem_after = psutil.Process().memory_info().rss / 1e6
        
        run_result = {
            "run": run + 1,
            "chunks": chunk_times,
            "ttfa": chunk_times[0]["latency"] if chunk_times else 0,
            "total_gen_time": t_total,
            "total_audio": total_audio,
            "overall_rtf": t_total / total_audio if total_audio > 0 else 0,
            "gaps": gaps,
            "mem_peak_mb": max(mem_before, mem_after),
            "mem_delta_mb": mem_after - mem_before,
        }
        
        # With parallel playback: how bad are gaps?
        if gaps:
            max_gap = max(g["gap_parallel_best"] for g in gaps)
            print(f"    Max gap (parallel playback): {max_gap:.2f}s", flush=True)
        
        results.append(run_result)
    
    return results


# Run benchmarks
import psutil

# 1. Non-streaming (fallback) — full text, single gen
print(f"\n{'='*70}")
print(f"NON-STREAMING (FALLBACK) — {len(LONG_TEXT)} chars")
print(f"{'='*70}")
ns_times = []
for run in range(5):
    t0 = time.time()
    wav, dur = tts([LONG_TEXT], style, total_step=STEPS, speed=SPEED, lang=LANG)
    t = time.time() - t0
    audio_dur = float(dur[0])
    ns_times.append({"run": run+1, "gen_time": t, "audio_dur": audio_dur, "rtf": t/audio_dur})
    print(f"  Run {run+1}: {t:.2f}s gen, {audio_dur:.2f}s audio, RTF={t/audio_dur:.3f}", flush=True)

# 2. Streaming benchmark
results = run_benchmark(LONG_TEXT, "TEXT LUNG", n_runs=5)
results_short = run_benchmark(SHORT_TEXT, "TEXT SCURT", n_runs=3)

# ========== SUMMARY ==========
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

print(f"\n--- NON-STREAMING (FALLBACK) ---")
ns_gen = [r["gen_time"] for r in ns_times]
ns_audio = [r["audio_dur"] for r in ns_times]
ns_rtf = [r["rtf"] for r in ns_times]
ns_gen_sorted = sorted(ns_gen)
print(f"Gen time: median={np.median(ns_gen):.2f}s, p95={np.percentile(ns_gen, 95):.2f}s")
print(f"Audio: {np.median(ns_audio):.2f}s")
print(f"RTF: median={np.median(ns_rtf):.3f}, p95={np.percentile(ns_rtf, 95):.3f}")

print(f"\n--- STREAMING (TEXT LUNG, {len(results)} runs) ---")
ttfas = [r["ttfa"] for r in results]
rtfs = [r["overall_rtf"] for r in results]
mems = [r["mem_peak_mb"] for r in results]

print(f"TTFA (Time to first audio): median={np.median(ttfas):.2f}s, p95={np.percentile(ttfas, 95):.2f}s")
print(f"Overall RTF: median={np.median(rtfs):.3f}, p95={np.percentile(rtfs, 95):.3f}")
print(f"Peak RAM: median={np.median(mems):.1f}MB, p95={np.percentile(mems, 95):.1f}MB")

# Gap analysis
all_gaps = []
for r in results:
    for g in r["gaps"]:
        all_gaps.append(g["gap_parallel_best"])
if all_gaps:
    print(f"Gaps (parallel playback): median={np.median(all_gaps):.2f}s, max={max(all_gaps):.2f}s")
    print(f"Gaps > 0 (audible pause): {sum(1 for g in all_gaps if g > 0)}/{len(all_gaps)}")
    positive_gaps = [g for g in all_gaps if g > 0]
    if positive_gaps:
        print(f"  Positive gaps mean: {np.mean(positive_gaps):.2f}s")
    negative_gaps = [g for g in all_gaps if g <= 0]
    if negative_gaps:
        print(f"  Non-positive gaps (smooth playback): {len(negative_gaps)}/{len(all_gaps)}")

print(f"\n{'='*70}")
print(f"VERDICT")
print(f"{'='*70}")
median_rtf = np.median(rtfs)
median_ttfa = np.median(ttfas)
median_ns = np.median(ns_rtf)

print(f"Streaming TTFA: {median_ttfa:.1f}s vs Non-streaming: {np.median(ns_gen):.1f}s")
print(f"Improvement: {(np.median(ns_gen) - median_ttfa) / np.median(ns_gen) * 100:.0f}% faster to first audio")
if median_rtf > 1.0:
    print(f"⚠️ RTF={median_rtf:.2f} > 1.0 — GENERAREA E MAI LENTĂ DECât REDAREA")
    print(f"   Gaps between chunks sunt INEVITABILE cu RTF > 1")
    if all_gaps and max(all_gaps) > 0:
        print(f"   Gap maxim: {max(all_gaps):.1f}s — pauză sesizabilă între chunk-uri")
else:
    print(f"✅ RTF={median_rtf:.2f} < 1.0 — generarea e suficient de rapidă")
