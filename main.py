# main.py

import os, re, json, tempfile, urllib.parse, time, asyncio
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp, pdfplumber, google.generativeai as genai, requests
from dotenv import load_dotenv

# ---------------- Global Safe-Mode Timeouts ----------------
REQUEST_TIMEOUT = 30
YTDLP_SOCKET_TIMEOUT = 15
YTDLP_RETRIES = 1
HLS_SEGMENT_LIMIT = 50

# ---------------- Setup ----------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
YTDLP_COOKIES = os.getenv("YTDLP_COOKIES")
if YTDLP_COOKIES and not os.path.exists(YTDLP_COOKIES):
    print(f"[SmartSummarizer] Warning: cookie file not found at {YTDLP_COOKIES}")
    YTDLP_COOKIES = None

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        print("Warning: genai.configure() failed:", e)

app = FastAPI(title="Smart Summarizer (Fast Async)", version="1.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ThreadPoolExecutor for blocking synchronous calls
executor = ThreadPoolExecutor(max_workers=4)

# ---------------- Utilities ----------------
def log(s: str): 
    print(f"[SmartSummarizer] {s}")

def _safe_gemini_text(resp) -> str:
    if not resp: return ""
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip(): return text.strip()
    if isinstance(resp, dict):
        if resp.get("text"): return resp["text"].strip()
        if resp.get("candidates"):
            try:
                cand = resp["candidates"][0]
                txt = cand.get("content", {}).get("text") or cand.get("text")
                if txt: return txt.strip()
            except Exception: pass
    return str(resp).strip() if resp else ""

async def run_blocking(func, *args, **kwargs):
    """Helper to run blocking functions in the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

# ---------------- YouTube Helpers (Blocking -> Wrapped) ----------------
def extract_video_id(url: str) -> str:
    pats = [r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&\?#]|$)", r"youtu\.be\/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    m2 = re.search(r"([0-9A-Za-z_-]{11})", url)
    if m2: return m2.group(1)
    raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

def _sync_try_transcript_api(video_id: str, video_url: Optional[str] = None) -> Optional[str]:
    """
    Synchronous implementation of transcript fetching.
    Optimized: Removed slow yt_dlp pre-check.
    """
    try:
        log("Attempting YouTubeTranscriptApi (optimized universal)...")
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        english_variants = [
            "English", "en", "en-us", "en-gb", "en-in",
            "English (auto-generated)", "English (US)", "English (UK)", "English (India)"
        ]

        english_matches, others = [], []
        for t in transcripts:
            ln = t.language.lower()
            lc = t.language_code.lower()
            if any(v in ln or v == lc for v in english_variants):
                english_matches.append(t)
            else:
                others.append(t)

        for group, label in [(english_matches, "English"), (others, "non-English")]:
            for t in group:
                try:
                    fetched = t.fetch()
                    if fetched:
                        text = " ".join(seg.get("text", "") for seg in fetched if seg.get("text"))
                        if text.strip():
                            log(f"Fetched {label} transcript ({t.language}, {t.language_code})")
                            return text
                except Exception as e:
                    if "no element found" in str(e).lower():
                        log("Transcript XML empty (likely HLS) — skipping.")
                        return None
                    else:
                        log(f"{label} transcript fetch failed: {e}")

    except Exception as e:
        log(f"YouTubeTranscriptApi optimized fetch failed: {e}")

    return None

async def try_transcript_api(video_id: str, video_url: Optional[str] = None) -> Optional[str]:
    return await run_blocking(_sync_try_transcript_api, video_id, video_url)


# ---------------- yt-dlp Wrapper ----------------
def _extract_with_stable_client(video_url: str, download: bool, extra_opts: Optional[dict] = None):
    ydl_opts = {
        "quiet": True, "no_warnings": True, "skip_download": not download,
        "extractor_args": {"youtube": {"player_client": ["default"], "skip_bad_formats": ["True"]}},
        "retries": YTDLP_RETRIES, "socket_timeout": YTDLP_SOCKET_TIMEOUT,
    }
    if YTDLP_COOKIES: ydl_opts["cookiefile"] = YTDLP_COOKIES
    if extra_opts: ydl_opts.update(extra_opts)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(video_url, download=download)
    except Exception:
        ydl_opts["extractor_args"]["youtube"]["player_client"] = ["android"]
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(video_url, download=download)

def _subtitle_to_plain(s: str) -> str:
    s = s.strip()
    if not s: return ""
    try:
        data = json.loads(s)
        segs = []
        if isinstance(data, dict) and "events" in data:
            for ev in data["events"]:
                for sg in ev.get("segs") or []:
                    t = sg.get("utf8", "").strip()
                    if t: segs.append(t)
        if segs: return " ".join(segs)
    except Exception:
        pass
    lines = [ln.strip() for ln in s.splitlines() if ln and not re.match(r"^(\d+|WEBVTT|-->)", ln)]
    return " ".join(lines)

def _sync_try_ytdlp_subtitles(video_url: str) -> Optional[str]:
    """Prefer English captions first, otherwise use any available language."""
    log("Attempting yt_dlp subtitle extraction (enhanced)...")
    start = time.time()

    def is_english_label(label: str) -> bool:
        if not label:
            return False
        lbl = label.lower()
        return (
            "english" in lbl or
            re.match(r"^en([\-_][a-z]+)?$", lbl) is not None or
            lbl.startswith("en") or
            lbl == "eng"
        )

    try:
        info = _extract_with_stable_client(video_url, download=False)
        subs = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}

        all_tracks = {**subs, **auto}

        english_candidates = []
        other_candidates = []

        # Split tracks into English vs non-English
        for lang, items in all_tracks.items():
            if not isinstance(items, list):
                continue
            if is_english_label(lang):
                english_candidates.extend(items)
            else:
                other_candidates.extend(items)

        # Priority: English -> other languages -> anything
        if english_candidates:
            candidates = english_candidates
        elif other_candidates:
            candidates = other_candidates
        else:
            candidates = []
            for v in all_tracks.values():
                if isinstance(v, list):
                    candidates.extend(v)

        for cand in candidates:
            if time.time() - start > REQUEST_TIMEOUT:
                break
            url = cand.get("url")
            if not url:
                continue
            try:
                r = requests.get(url, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200 and r.text.strip():
                    text = r.text

                    # HLS playlist?
                    if text.lstrip().startswith("#EXTM3U"):
                        log("Detected HLS (.m3u8) subtitle playlist; fetching segments...")
                        lines = text.splitlines()
                        segs = [
                            urllib.parse.urljoin(url, ln.strip())
                            for ln in lines if ln and not ln.startswith("#")
                        ]
                        collected = []
                        for seg in segs[:HLS_SEGMENT_LIMIT]:
                            try:
                                rs = requests.get(seg, timeout=REQUEST_TIMEOUT)
                                if rs.status_code == 200:
                                    content = rs.content.decode("utf-8", errors="ignore")
                                    cleaned = _subtitle_to_plain(content)
                                    if cleaned.strip():
                                        collected.append(cleaned)
                            except Exception as e:
                                log(f"Segment fetch failed: {e}")
                        if collected:
                            log(f"Fetched {len(collected)} HLS subtitle segments.")
                            return " ".join(collected)
                        continue

                    # Normal VTT/SRT text
                    cleaned = _subtitle_to_plain(text)
                    if cleaned.strip():
                        log("yt_dlp subtitle fetch succeeded.")
                        return cleaned
            except Exception as e:
                log(f"Subtitle fetch failed: {e}")

        log("No usable subtitles found via yt_dlp.")
    except Exception as e:
        log(f"yt_dlp subtitle extraction failed: {e}")

    return None

async def try_ytdlp_subtitles(video_url: str) -> Optional[str]:
    return await run_blocking(_sync_try_ytdlp_subtitles, video_url)


# ---------------- Master Transcript Logic ----------------
async def extract_transcript_from_youtube(video_url: str) -> str:
    log(f"extract_transcript_from_youtube START for {video_url}")
    vid = extract_video_id(video_url)
    
    # Try official API first
    transcript = await try_transcript_api(vid, video_url)
    
    # Fallback to yt-dlp subtitles
    if not transcript:
        transcript = await try_ytdlp_subtitles(video_url)
        
    # Fallback to metadata
    if not transcript:
        log("No subtitles available — using title and description as fallback.")
        try:
            info = await run_blocking(_extract_with_stable_client, video_url, download=False)
            title = info.get("title", "")
            desc = info.get("description", "")
            transcript = f"Title: {title}\n\nDescription:\n{desc}"
        except Exception as e:
            log(f"Metadata fetch failed: {e}")
            transcript = ""

    if transcript and transcript.strip():
        return transcript
        
    # Final desperate fallback
    try:
        info = await run_blocking(_extract_with_stable_client, video_url, download=False)
        return f"Title: {info.get('title','')}\n\nDescription:\n{info.get('description','')}"
    except Exception:
        pass
    raise HTTPException(status_code=400, detail="Transcript unavailable.")

# ---------------- PDF Extraction ----------------
def _sync_extract_pdf_text(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

async def extract_pdf_text(pdf_path: str) -> str:
    return await run_blocking(_sync_extract_pdf_text, pdf_path)

# ---------------- Summarization ----------------
async def summarize_via_gemini(text: str, summary_type: str = "short",
                         bullet_count: Optional[int] = None, target_lang: str = "en") -> str:

    if not text:
        return "No text available to summarize."
    if not GEMINI_KEY:
        return text[:800] + "..." * (len(text) > 800)

    summary_type = (summary_type or "short").strip().lower()
    if summary_type not in {"short", "bullet", "detailed"}:
        log(f"Unknown summary_type '{summary_type}', defaulting to 'short'")
        summary_type = "short"

    model = genai.GenerativeModel("gemini-2.0-flash")

    # -------- Optimized Chunking --------
    text = text.strip()
    text_len = len(text)

    # Increased to 3M chars to leverage Gemini 2.0 Flash context window
    MAX_SAFE_CHUNK = 3000000 
    overlap = 500
    max_chunks = 3 

    if text_len <= MAX_SAFE_CHUNK:
        chunks = [text]
    else:
        chunks = []
        step = MAX_SAFE_CHUNK - overlap
        for i in range(0, text_len, step):
            chunk = text[i:i + MAX_SAFE_CHUNK]
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break

    log(f"[Chunker v1.3.1] Processing {len(chunks)} chunk(s) (~{MAX_SAFE_CHUNK} chars each, total={text_len})")

    # -------- Helper for parallel execution --------
    async def process_chunk(chunk, idx):
        try:
            if summary_type == "bullet":
                bc = int(bullet_count) if bullet_count else 10
                prompt = f"""You are a professional expert summarizer. Extract exactly {bc} key bullet points from this text.

Requirements:
- Each bullet must be a distinct, independent fact or concept
- Bullets should be concise but complete (1-2 sentences max)
- Organize logically by topic or sequence
- Use clear, professional language
- No bullets should repeat or overlap
- Format: Use '-' at the start of each bullet point

Content:
{chunk}"""
                max_tokens = 1200

            elif summary_type == "detailed":
                prompt = f"""Create an extremely detailed and comprehensive summary with thorough explanations for every point.

Format requirements:
- Use ## for main sections and ### for subsections
- Under each section, provide detailed bullet points with comprehensive explanations
- Each bullet point should be 2-3 sentences explaining the concept thoroughly
- Include practical examples, implications, and context for each point
- Organize related concepts together logically
- Maintain professional academic tone throughout
- Ensure all important details and nuances are captured
- Include at least 5-7 major topics with multiple detailed points each

Content:
{chunk}"""
                max_tokens = 3000

            else:  # short
                prompt = f"""Write a concise professional summary in 3-4 clear paragraphs.

Guidelines:
- Each paragraph should focus on one main idea
- Use clear transitions between paragraphs
- Keep language formal and professional
- Highlight key points and conclusions
- Avoid repetition

Content:
{chunk}"""
                max_tokens = 1200

            # Async generation
            resp = await model.generate_content_async(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": max_tokens, "top_p": 0.9}
            )
            result = _safe_gemini_text(resp)
            if result.strip():
                log(f"✔ Chunk {idx} completed ({len(chunk)} chars)")
                return result
            else:
                log(f"⚠ Empty response for chunk {idx}")
                return ""
        except Exception as e:
            log(f"Error processing chunk {idx}: {e}")
            return ""

    # -------- Parallel Execution --------
    tasks = [process_chunk(chunk, i+1) for i, chunk in enumerate(chunks)]
    partials = await asyncio.gather(*tasks)
    partials = [p for p in partials if p] # Filter empty results

    if not partials:
        return text[:1500] + "..."

    # -------- Skip merge for single chunk --------
    if len(partials) == 1:
        log("✅ Summary generated successfully")
        return format_summary_output(partials[0], summary_type)

    # -------- Merge multiple chunks --------
    combined = "\n\n".join(partials)
    log("Merging partial summaries...")

    try:
        if summary_type == "short":
            merge_prompt = f"""Combine these partial summaries into ONE cohesive professional summary (3-4 paragraphs).
- Remove any duplicate information
- Maintain logical flow
- Keep professional tone

Partial summaries:
{combined}"""
        elif summary_type == "bullet":
            bc = int(bullet_count) if bullet_count else 10
            merge_prompt = f"""Combine and deduplicate to produce exactly {bc} key bullet points.
- Each bullet must be unique and valuable
- Remove redundancy
- Keep format simple with '-' prefix

Partial summaries:
{combined}"""
        else:  # detailed
            merge_prompt = f"""Combine these sections into ONE comprehensive, well-organized summary.
- Keep ## for main sections and ### for subsections
- Remove duplicates while preserving all important details
- Ensure each point has thorough, detailed explanations (2-3 sentences per point)
- Maintain clear structure with comprehensive bullet points under each section
- Include all nuances, implications, and context

Partial summaries:
{combined}"""

        resp = await model.generate_content_async(
            merge_prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 3500, "top_p": 0.9}
        )
        final_summary = _safe_gemini_text(resp).strip()

        if final_summary:
            log("✅ Final summary merged and optimized")
            return format_summary_output(final_summary, summary_type)

    except Exception as e:
        log(f"Merge error: {e}")
        return format_summary_output("\n\n".join(partials), summary_type)

    return format_summary_output("\n\n".join(partials[:3]), summary_type)

# ----------- HTML Styling Formatter -----------
def format_summary_output(text: str, summary_type: str) -> str:
    text = text.strip()
    if not text:
        return ""

    # --- Convert bold (**text**) to <b>text</b> ---
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    if summary_type == "bullet":
        # Normalize all bullet markers first
        text = text.replace("•", "- ").replace("*", "- ")

        # Fix inline bullets like "- a - b - c"
        text = re.sub(r"\s+-\s+", "\n- ", text)

        # Ensure each bullet starts on a new line (but only when needed)
        cleaned = []
        for ln in text.splitlines():
            ln = ln.strip()
            if ln.startswith("-"):
                cleaned.append(ln)
            else:
                if ln:
                    cleaned.append(ln)

        # Strict bullet extraction
        bullets = []
        for ln in cleaned:
            if ln.startswith("-"):
                b = ln[1:].strip()
                if b:
                    bullets.append(b)

        # Deduplicate while preserving order
        final_list = []
        seen = set()
        for b in bullets:
            if b not in seen:
                seen.add(b)
                final_list.append(b)

        # Convert to clean HTML
        html = "<ul>\n" + "\n".join(f"<li>{b}</li>" for b in final_list) + "\n</ul>"
        return html

    elif summary_type in ["detailed"]:
        # Convert Markdown headers and bullets to HTML
        text = re.sub(r"^###\s*(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^##\s*(.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)

        # Additional fixes — catch missing spaces
        text = re.sub(r"^###(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^##(.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)

        # Convert bolded colon lines (**Term:**) into section headers
        text = re.sub(r"<b>([^<:]+:)</b>", r"<h4>\1</h4>", text)

        # Convert bullet lines
        lines = [ln.strip() for ln in text.splitlines()]
        formatted = []
        for ln in lines:
            if re.match(r"^[-*•]", ln):
                ln = re.sub(r"^[-*•]\s*", "", ln)
                formatted.append(f"<li>{ln}</li>")
            else:
                formatted.append(ln)
        html = "\n".join(formatted)
        html = re.sub(r"(<li>.+?</li>)+", lambda m: f"<ul>{m.group(0)}</ul>", html)
        return html

    else:  # short summary
        paras = [f"<p>{p.strip()}</p>" for p in text.split("\n\n") if p.strip()]
        return "\n".join(paras)
    

# ---------------- Endpoints ----------------
@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")

@app.get("/styles.css")
async def serve_css():
    return FileResponse(Path(__file__).parent / "styles.css", media_type="text/css")

@app.get("/script.js")
async def serve_js():
    return FileResponse(Path(__file__).parent / "script.js", media_type="application/javascript")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.3.1",
        "gemini_configured": bool(GEMINI_KEY)
    }

@app.post("/summarize/youtube")
async def summarize_youtube(video_url: str = Form(...),
                            summary_type: str = Form("short"),
                            bullet_count: Optional[int] = Form(None),
                            target_lang: str = Form("en")):
    try:
        log(f"Processing YouTube: {video_url[:50]}... ({summary_type})")
        transcript = await extract_transcript_from_youtube(video_url)
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="No transcript found.")
        
        log("Generating summary...")
        final = await summarize_via_gemini(transcript, summary_type, bullet_count, "en")
        
        return {
            "success": True, 
            "summary": final, 
            "video_url": video_url,
            "summary_type": summary_type,
            "transcript_length": len(transcript)
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in YouTube summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/pdf")
async def summarize_pdf(file: UploadFile = File(...),
                        summary_type: str = Form("short"),
                        bullet_count: Optional[int] = Form(None),
                        target_lang: str = Form("en")):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        await run_blocking(tmp.write, content)
        tmp.close()
        
        log(f"Processing PDF: {file.filename} ({summary_type})")
        text = await extract_pdf_text(tmp.name)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        
        log("Generating summary...")
        final = await summarize_via_gemini(text, summary_type, bullet_count, "en")
        
        return {
            "success": True, 
            "summary": final, 
            "filename": file.filename,
            "summary_type": summary_type,
            "text_length": len(text)
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in PDF summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
