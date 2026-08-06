"""
next-read v10.5: blurb-first, with optional casual shares.

CHANGES FROM v10.4 (cost-reduction pass — quality-neutral):
  - CACHE_FILE prefers /data/cache.json (HF Spaces persistent storage)
    when /data exists and is writable; falls back to local cache.json.
    Enable "Persistent storage" in the Space settings for this to stick
    across restarts.
  - Gather cache entries now carry a "cached_at" timestamp. Fresh hits
    (< CACHE_MAX_AGE_DAYS, default 30) are returned directly WITHOUT
    re-running the full multi-call gather on top. Stale hits keep the
    old top-up behavior. Override per call with force_refresh=True.
  - Token usage tracking: every OpenAI call records input/output/cached
    token counts; per-query totals + estimated cost land in telemetry
    (metrics.json) so cost per query is measurable, not guessed.
  - Prompt restructure for OpenAI automatic prompt caching: the long
    static instruction blocks in gather/classify prompts now come FIRST
    and the per-name/per-book variables come LAST, so the shared prefix
    is cache-eligible (~50-75% off those input tokens on cache hits).
    Also sets prompt_cache_key per call type to improve hit routing.

CHANGES FROM v10.3:
  - Strip leading "the " in _normalize_title to fix dedup misses
    (e.g. "The Pyramid of Lies" and "Pyramid of Lies" now merge correctly).
  - Byline-as-quote filter in rank(): skip evidence quotes that are just
    the endorser's own name/attribution line.
  - Removed unused `base_name` variable in _gather_blurbs.
"""

import json
import os
import re
import time
import hashlib
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
MODEL = "gpt-4.1"

BLURB_RUNS = 10
CASUAL_RUNS = 3
MAX_WORKERS = 12

MIN_SAME_PERSON_CONFIDENCE = 0.5

BLURB_SIGNALS = {"blurb", "foreword", "introduction", "jacket_quote", "praise_page"}
CASUAL_SIGNALS = {"tweet", "blog_post", "substack", "instagram",
                  "podcast_moment", "interview_moment", "social_post"}

KNOWN_SIGNALS = BLURB_SIGNALS | CASUAL_SIGNALS | {"not_a_blurb", "not_casual", "unknown"}

def _pick_cache_path():
    """Prefer HF Spaces persistent storage (/data) when available, so the
    cache survives Space restarts/rebuilds. Falls back to the app dir
    (ephemeral on Spaces without persistent storage). Seeds /data from a
    repo-committed cache.json on first run so prior entries carry over."""
    data_dir = "/data"
    if os.path.isdir(data_dir) and os.access(data_dir, os.W_OK):
        persistent = os.path.join(data_dir, "cache.json")
        if not os.path.exists(persistent) and os.path.exists("cache.json"):
            try:
                import shutil
                shutil.copy("cache.json", persistent)
                print(f"  [cache] seeded {persistent} from repo cache.json")
            except OSError as e:
                print(f"  [warning] could not seed persistent cache: {e}")
        return persistent
    return "cache.json"


CACHE_FILE = _pick_cache_path()
METRICS_FILE = "metrics.json"
CACHE_MAX_AGE_DAYS = 30  # gather results younger than this skip re-gather
_cache = None
_cache_lock = threading.Lock()
_metrics_lock = threading.Lock()

# --- token usage tracking (thread-safe, reset per query) ---
_usage_lock = threading.Lock()
_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
          "api_calls": 0, "web_search_calls": 0}

# $/1M tokens for gpt-4.1; update if MODEL changes. Cached input is 75% off.
PRICE_INPUT = 2.00
PRICE_CACHED_INPUT = 0.50
PRICE_OUTPUT = 8.00
PRICE_WEB_SEARCH_PER_CALL = 0.01  # $10 per 1k tool calls


def _record_usage(response):
    """Accumulate token usage from a Responses API result."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    cached = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    searches = 0
    for item in (getattr(response, "output", None) or []):
        if getattr(item, "type", "") == "web_search_call":
            searches += 1
    with _usage_lock:
        _usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        _usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        _usage["cached_tokens"] += cached
        _usage["api_calls"] += 1
        _usage["web_search_calls"] += searches


def _usage_snapshot_and_reset():
    with _usage_lock:
        snap = dict(_usage)
        for k in _usage:
            _usage[k] = 0
    uncached_input = max(0, snap["input_tokens"] - snap["cached_tokens"])
    snap["est_cost_usd"] = round(
        uncached_input / 1e6 * PRICE_INPUT
        + snap["cached_tokens"] / 1e6 * PRICE_CACHED_INPUT
        + snap["output_tokens"] / 1e6 * PRICE_OUTPUT
        + snap["web_search_calls"] * PRICE_WEB_SEARCH_PER_CALL, 4)
    return snap


# ============================================================
# AUTHOR FILTER
# ============================================================

def _strip_disambiguator(name):
    return re.sub(r"\s*\([^)]*\)", "", name or "").strip()


def _name_tokens(name):
    base = _strip_disambiguator(name).lower()
    base = re.sub(r"\b[a-z]\.\s*", " ", base)
    tokens = re.split(r"\s+", base.strip())
    filtered = [t for t in tokens if len(t) > 2]
    # Short surnames ("Ma", "Ng", "Xu") would otherwise be dropped, leaving
    # <2 tokens and silently disabling the author filter for that person.
    if len(filtered) < 2:
        filtered = [t for t in tokens if t]
    return filtered


def _is_author_of_book(endorser_name, book_author):
    if not book_author or not endorser_name:
        return False
    endorser_tokens = _name_tokens(endorser_name)
    if len(endorser_tokens) < 2:
        return False
    book_author_lower = (book_author or "").lower()
    first_name = endorser_tokens[0]
    last_name = endorser_tokens[-1]
    if first_name in book_author_lower and last_name in book_author_lower:
        return True
    base_full = _strip_disambiguator(endorser_name).lower()
    if base_full and base_full in book_author_lower:
        return True
    return False


# ============================================================
# TELEMETRY
# ============================================================

def _log_metric(record):
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with _metrics_lock:
        try:
            with open(METRICS_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"  [warning] failed to write metrics: {e}")


# ============================================================
# CACHE
# ============================================================

def _load_cache():
    global _cache
    if _cache is None:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    _cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache_locked():
    if _cache is None:
        return
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_cache, f, indent=2)
    except OSError:
        pass


def _cache_key(prefix, *args):
    payload = json.dumps([prefix, args], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cache_get(prefix, *args):
    with _cache_lock:
        return _load_cache().get(_cache_key(prefix, *args))


def _cache_set(prefix, value, *args):
    with _cache_lock:
        cache = _load_cache()
        cache[_cache_key(prefix, *args)] = value
        _save_cache_locked()


def _cache_delete(prefix, *args):
    with _cache_lock:
        cache = _load_cache()
        key = _cache_key(prefix, *args)
        if key in cache:
            del cache[key]
            _save_cache_locked()
            return True
        return False


# ============================================================
# DEDUP
# ============================================================

def _normalize_title(title):
    title = (title or "").lower().strip()
    title = re.sub(r"^the\s+", "", title)  # strip leading "the " for dedup
    title = re.sub(r"\s*\([^)]*\)\s*", " ", title)
    title = title.split(":")[0].split(" - ")[0].split(" — ")[0]
    title = re.sub(r"[^a-z0-9 ]+", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:50]


def _book_dedup_key(book):
    return _normalize_title(book.get("title", ""))


def _author_score(author):
    if not author:
        return 0
    a = author.lower()
    if "unknown" in a:
        return 0
    if "?" in a:
        return 1
    return len(author)


def _merge_book_into(books_dict, book):
    key = _book_dedup_key(book)
    if not key.strip():
        return
    existing = books_dict.get(key)
    if existing is None:
        if "signal_types" not in book:
            book["signal_types"] = [book.get("signal_type", "unknown")]
        books_dict[key] = book
        return

    incoming = book.get("signal_types") or [book.get("signal_type", "unknown")]
    existing_signals = existing.get("signal_types") or [existing.get("signal_type", "unknown")]
    existing["signal_types"] = list(dict.fromkeys(existing_signals + incoming))

    if len(book.get("title", "")) > len(existing.get("title", "")):
        existing["title"] = book["title"]

    if _author_score(book.get("author")) > _author_score(existing.get("author")):
        existing["author"] = book["author"]

    for k, v in book.items():
        if k in ("signal_type", "signal_types", "title", "author"):
            continue
        if v and not existing.get(k):
            existing[k] = v


def _clean_url(url):
    if not url:
        return ""
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return ""
    return url


# ============================================================
# OpenAI wrappers
# ============================================================

def _call_with_search(prompt, max_output_tokens=6000, max_retries=2, temperature=0.7,
                      cache_key=None):
    for attempt in range(max_retries + 1):
        try:
            kwargs = {}
            if cache_key:
                kwargs["prompt_cache_key"] = cache_key
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                tools=[{"type": "web_search"}],
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                **kwargs,
            )
            _record_usage(response)
            text = response.output_text or ""
            if not text:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                print(f"  [debug] empty response after {max_retries + 1} attempts")
            return text
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(f"  [error] OpenAI API call failed: {e}")
            return ""
    return ""


def _call_plain(prompt, max_output_tokens=600, cache_key=None):
    try:
        kwargs = {}
        if cache_key:
            kwargs["prompt_cache_key"] = cache_key
        response = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=0,
            **kwargs,
        )
        _record_usage(response)
        return response.output_text or ""
    except Exception as e:
        print(f"  [warning] plain LLM call failed: {e}")
        return ""


# ============================================================
# DISAMBIGUATION
# ============================================================

def disambiguate_person(person_name):
    cached = _cache_get("disambig_v3", person_name)
    if cached is not None:
        return cached

    prompt = f"""A user wants to find books that {person_name} has personally endorsed (blurbs, forewords, etc).

If "{person_name}" is a common name shared by multiple notable people, identify them.

Pick the SINGLE person most likely to publicly endorse or recommend books. Priority order:
1. Authors, journalists, academics
2. Business executives, investors, public intellectuals
3. Athletes, artists, musicians, chefs, public figures
4. Anyone else notable

In the disambiguator (the part in parens after their name), pack in their most-famous work, role, or area of expertise — anything that would help someone reading "Ben Smith (X)" instantly know which person we mean AND what kind of books they'd be most likely to blurb. Examples:
- Ken Griffin (Citadel founder)
- Bradley Hope (journalist, co-author of 'Billion Dollar Whale')
- John Carreyrou (journalist, author of 'Bad Blood')
- Stephen King (horror novelist, author of 'The Shining')
- Yo-Yo Ma (cellist)
- Ina Garten (cookbook author and TV host)
- LeBron James (NBA player and producer)
- Anthony Bourdain (chef, author of 'Kitchen Confidential')

The disambiguator should be 3–8 words. Include their best-known work in single quotes if relevant. Cover ANY beat — sports, art, fiction, food, science, etc. — not just business/finance.

Return ONLY valid JSON, no markdown:
{{
  "primary": "Full name (rich disambiguator with beat or famous work)",
  "alternatives": ["Other notable people with this name (brief disambig each)"],
  "is_ambiguous": true or false
}}

Even if unambiguous, ALWAYS add a rich disambiguator.

Name: {person_name}"""

    text = _call_plain(prompt, max_output_tokens=500)
    data = _extract_json(text)
    primary = data.get("primary")
    if not primary:
        return {"primary": person_name, "alternatives": [], "is_ambiguous": False}

    result = {
        "primary": primary,
        "alternatives": data.get("alternatives", []),
        "is_ambiguous": bool(data.get("is_ambiguous")),
    }
    _cache_set("disambig_v3", result, person_name)
    return result


# ============================================================
# BLURB GATHER
# ============================================================

def name_variants(full_name):
    cached = _cache_get("variants_v1", full_name)
    if cached is not None:
        return cached

    prompt = f"""Generate plausible name variants for this person, for web search.

"{full_name}"

If the input has a disambiguator in parens, preserve it across variants so searches stay targeted on the right person.

Return common variations: formal name, nicknames, common shortenings, middle initial variations, etc.

Examples:
- "Michael Bloomberg" -> ["Michael Bloomberg", "Mike Bloomberg", "Michael R. Bloomberg"]
- "Ken Griffin (Citadel founder)" -> ["Ken Griffin (Citadel founder)", "Kenneth Griffin (Citadel founder)", "Kenneth C. Griffin (Citadel founder)"]
- "Bradley Hope (journalist, co-author of 'Billion Dollar Whale')" -> ["Bradley Hope (journalist, co-author of 'Billion Dollar Whale')", "Brad Hope (journalist, co-author of 'Billion Dollar Whale')", "Bradley P. Hope (journalist, co-author of 'Billion Dollar Whale')"]

Include 2–4 real, commonly-used variants. Always include the original input as the first item.

Return ONLY valid JSON (no markdown):
{{"variants": ["Original Name", "Variant 2", ...]}}"""

    text = _call_plain(prompt, max_output_tokens=400)
    data = _extract_json(text)
    variants = data.get("variants", []) if data else []
    if not variants:
        return [full_name]

    if full_name not in variants:
        variants = [full_name] + variants

    seen, deduped = set(), []
    for v in variants:
        key = (v or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(v)
    deduped = deduped[:3]
    _cache_set("variants_v1", deduped, full_name)
    return deduped


# Static instruction blocks kept byte-identical across every call so
# OpenAI's automatic prompt caching can reuse the prefix. The TARGET name
# goes at the very END of the prompt (variable content last = cacheable
# prefix). Same rules and wording as before, phrased around "the TARGET".

_BLURB_GATHER_STATIC = """You are searching for BACK-COVER BLURBS, FOREWORDS, INTRODUCTIONS, JACKET QUOTES, and PRAISE PAGE quotes by a TARGET person (named at the very end of this prompt). Endorsements that appear physically ON or IN a book.

WHAT COUNTS — and ONLY these things count:
- Back-cover blurb attributed to the TARGET by name
- Dust-jacket quote attributed to the TARGET
- "Praise for [book]" page quote attributed to the TARGET
- The TARGET wrote the foreword for a book
- The TARGET wrote the introduction for a book
- Amazon "Editorial Reviews" / publisher product page quoting the TARGET as endorser

WHAT DOES NOT COUNT (skip these always):
- The TARGET's tweets, blog posts, Substack, podcast mentions about books
- The TARGET's shareholder letters, annual letters, commencement speeches
- Books on the TARGET's "recommended reading list" or "favorite books" (aggregator lists)
- Books the TARGET reviewed in a publication
- Books that mention or profile the TARGET (no quote FROM them)
- Books authored or co-authored by the TARGET
- Biographies of the TARGET

THE TEST: the TARGET's name and quoted endorsement text appears physically ON or IN the book — back cover, front cover, dust jacket, opening "praise for" page, retailer page's "Editorial Reviews" or "Praise" section, or foreword/introduction byline. ALL of these positions count equally. Do NOT downgrade or reject an endorsement just because you can't verify it's literally on the back cover specifically — if a publisher, retailer, or "praise for" page attributes a quote to the TARGET as an endorser of the book, that IS a blurb for our purposes. "Professional reviews" listed on a book's retail page (BetterWorldBooks, Amazon Editorial Reviews, Apple Books, etc.) attributed to the TARGET also count.

YOU MUST BE EXHAUSTIVE. The user cares MOST about not missing a real blurb. Cast a wide net. Search aggressively across many angles:
- "praise for [book]" + the TARGET's name
- "blurbed by [TARGET name]" / "endorsed by [TARGET name]"
- "foreword by [TARGET name]" / "introduction by [TARGET name]"
- Amazon editorial reviews mentioning the TARGET
- Publisher product pages quoting the TARGET
- Barnes & Noble book pages
- Google Books preview pages (back-matter section)
- Goodreads book pages quoting the TARGET
- BetterWorldBooks, Bookshop.org, Apple Books retailer pages
- Books in the TARGET's topic area — check their "praise" pages for the TARGET's name
- The TARGET's name + book titles related to their expertise

CRITICAL: also try TOPIC-SPECIFIC queries combining the TARGET's name with major subjects in their beat. For example, "[TARGET name] Musk book" or "[TARGET name] crypto book" or "[TARGET name] hedge fund book" — these surface specific blurb pages that pure "[TARGET name] blurb" queries miss. Think of the major subjects the TARGET covers and run a separate search for each.

If the TARGET has a disambiguator in parens, only include endorsements from THAT specific person.

Look back 20+ years. Aim for 10+ if they exist — but only include real blurbs.

Return ONLY valid JSON (no markdown, no trailing commas):
{
  "endorser": "<the TARGET's full name as given>",
  "books": [
    {
      "title": "Full Book Title Including Subtitle",
      "author": "Author Name (NOT the TARGET)",
      "year": 2023,
      "one_line": "brief description of the book",
      "signal_type": "blurb | foreword | introduction | jacket_quote | praise_page",
      "quote_snippet": "first 10-20 words of the TARGET's actual quote (very important)",
      "source_url": "URL where you saw this, only real https URLs",
      "notes": "any uncertainty"
    }
  ]
}

If you find none, return empty books array."""


def _gather_blurbs(name, run_idx):
    disambig_match = re.search(r"\(([^)]+)\)", name)
    disambig = disambig_match.group(1) if disambig_match else ""
    disambig_hint = (
        f"\nThis person is specifically {disambig}. When searching, pair the "
        f"name with terms from their profession/company so search results aren't "
        f"flooded with the wrong person."
        if disambig else ""
    )

    prompt = f"{_BLURB_GATHER_STATIC}\n\nTARGET: {name}{disambig_hint}"
    return _extract_json(_call_with_search(prompt, temperature=0.7,
                                           cache_key="gather-blurb"))


_CASUAL_GATHER_STATIC = """You are searching for CASUAL PERSONAL SHARES by a TARGET person (named at the very end of this prompt) where they recommended a book on a casual public channel — the kind of post where someone finishes a book and impulsively tells the world about it.

WHAT COUNTS:
- The TARGET's tweet / X post saying they read or loved a book
- The TARGET's Instagram post about a book
- The TARGET's personal blog post about a book they read
- The TARGET's Substack or newsletter post recommending a book
- A podcast moment where the TARGET spontaneously said "I just read [book], it's great"
- An on-the-record interview where the TARGET mentioned a book they personally loved
- The TARGET's LinkedIn post about a book
- The TARGET's Threads / Bluesky post about a book

THE TEST: Was this a casual, in-the-moment, personal recommendation in the TARGET's own voice on an informal channel? Did they sound like they finished reading and wanted to share?

WHAT DOES NOT COUNT (these are formal / curated documents, NOT casual shares):
- The TARGET's shareholder letters, annual letters, founder letters
- The TARGET's commencement speeches
- The TARGET's formal published reading lists
- Biographer's appendices listing what the TARGET reads
- Aggregator listicles ("10 books CEOs love")
- "Reading lists" compiled by third parties
- Back-cover blurbs / forewords (these are blurbs, a different category)
- Books that just mention or profile the TARGET
- Books authored by the TARGET

Be EXHAUSTIVE. Famous people who tweet about books (Bezos, Naval, Patrick Collison, etc.) often have many such posts. Aim for 10+ if they exist.

Search angles:
- "[TARGET name]" tweet book
- "[TARGET name]" Twitter / X book recommendation
- "[TARGET name]" Instagram book
- "[TARGET name]" Substack
- "[TARGET name]" blog book
- "[TARGET name]" podcast "just read" OR "loved"
- "[TARGET name]" interview favorite book

If the TARGET has a disambiguator in parens, only include the specific person.

Return ONLY valid JSON (no markdown, no trailing commas):
{
  "endorser": "<the TARGET's full name as given>",
  "books": [
    {
      "title": "Full Book Title Including Subtitle",
      "author": "Author Name (NOT the TARGET)",
      "year": 2023,
      "one_line": "brief description of the book",
      "signal_type": "tweet | blog_post | substack | instagram | podcast_moment | interview_moment | social_post",
      "quote_snippet": "first 10-20 words of what the TARGET said",
      "source_url": "URL of the tweet/post/transcript — required for casual shares",
      "notes": "any uncertainty"
    }
  ]
}

If you find none, return empty books array."""


def _gather_casual_shares(name, run_idx):
    disambig_match = re.search(r"\(([^)]+)\)", name)
    disambig = disambig_match.group(1) if disambig_match else ""
    disambig_hint = (
        f"\nThis person is specifically {disambig}. Include this context in "
        f"your searches to avoid the wrong person."
        if disambig else ""
    )

    prompt = f"{_CASUAL_GATHER_STATIC}\n\nTARGET: {name}{disambig_hint}"
    return _extract_json(_call_with_search(prompt, temperature=0.7,
                                           cache_key="gather-casual"))


# ============================================================
# GATHER ORCHESTRATION
# ============================================================

def gather_for_mode(name, mode, log=print, force_refresh=False):
    cache_prefix = f"gather_v10_6_{mode}"

    if force_refresh:
        if _cache_delete(cache_prefix, name):
            log(f"  [cache] wiped {cache_prefix} for {name}")

    cached = _cache_get(cache_prefix, name)

    # Freshness fast-path: if the cached gather is recent, serve it as-is
    # instead of re-running the full multi-call gather on top. Blurb
    # inventories change slowly; this is where most repeat-query cost dies.
    if cached and not force_refresh:
        cached_at = cached.get("cached_at")
        if cached_at:
            try:
                age_days = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(cached_at)
                ).days
            except ValueError:
                age_days = None
            if age_days is not None and age_days < CACHE_MAX_AGE_DAYS:
                log(f"  [cache] fresh hit ({age_days}d old, "
                    f"{len(cached.get('books', []))} candidates) — "
                    f"skipping re-gather")
                return cached

    if mode == "blurb":
        fn = _gather_blurbs
        runs = BLURB_RUNS
    else:
        fn = _gather_casual_shares
        runs = CASUAL_RUNS

    variants = name_variants(name)
    log(f"  Variants to search: {variants}")

    all_books = {}
    prior_count = 0
    if cached:
        for book in cached.get("books", []):
            if book.get("title", "").strip():
                _merge_book_into(all_books, book)
        prior_count = len(all_books)
        log(f"  [cache] starting with {prior_count} prior candidates, "
            f"running fresh gather to find more...")
    else:
        log(f"  [cache] no prior cache, running cold gather...")

    tasks = [(v, i) for v in variants for i in range(runs)]
    log(f"  Running {len(tasks)} parallel {mode} gather calls "
        f"({len(variants)} variants × {runs} runs)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fn, v, i) for (v, i) in tasks]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                for book in result.get("books", []):
                    if not book.get("title", "").strip():
                        continue
                    _merge_book_into(all_books, book)
            except Exception as e:
                log(f"  [warning] {mode} run failed: {e}")

    new_count = len(all_books) - prior_count
    if cached and new_count > 0:
        log(f"  [cache] added {new_count} new candidates "
            f"(total now {len(all_books)})")
    elif cached and new_count == 0:
        log(f"  [cache] no new candidates this run "
            f"(still {len(all_books)} total)")

    final = {
        "endorser": name,
        "mode": mode,
        "variants_searched": variants,
        "gather_runs": len(tasks),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "books": list(all_books.values()),
    }
    log(f"  Found {len(final['books'])} unique candidates "
        f"({prior_count} cached + {new_count} new)")

    if final["books"]:
        _cache_set(cache_prefix, final, name)
    return final


# ============================================================
# CLASSIFY
# ============================================================

# Static verifier instructions first (cache-eligible prefix); the specific
# book/endorser under review goes at the END of the prompt.
_CLASSIFY_STATIC = """You are the VERIFIER for the next-read app. A previous gather step has already found a likely endorsement match (given at the very end of this prompt).

Your job is NOT to independently re-prove the endorsement. The gather already did that.
Your job is ONLY to catch a few specific kinds of false positives.

DEFAULT BEHAVIOR: verified=true.

Search the web briefly to check for these RED FLAGS. Set verified=false ONLY if you find clear evidence of one:

RED FLAG 1: The endorser is the AUTHOR or CO-AUTHOR of this book.
  -> reason: "author of book"

RED FLAG 2: The endorsement is from a DIFFERENT person with the same name (different Ken Griffin, different Bradley Hope, etc.). Use the disambiguator in the endorser's name to tell which person we mean.
  -> reason: "different person with same name"

RED FLAG 3: The book is BY OR ABOUT the endorser (biography of them, memoir by them, book with their name in the title that they didn't blurb).
  -> reason: "book is by/about the person"

DO NOT reject just because:
- You couldn't independently find the blurb online (search results vary)
- The blurb is hard to verify
- You aren't 100% sure it's a "formal" blurb for the mode

Depending on the mode given at the end, the signal_type should be one of:
  blurb mode: blurb, foreword, introduction, jacket_quote, praise_page
  casual mode: tweet, blog_post, substack, instagram, podcast_moment, interview_moment, social_post

If you can't tell, return the prior tag from the gather step.

Return ONLY valid JSON:
{
  "verified": true,
  "signal_type": "<one of the mode's signal types>",
  "is_author": false,
  "same_person_confidence": 1.0,
  "source_url": "URL where you found evidence, or empty",
  "quote": "exact quoted text if found, or empty",
  "reason": "brief note, or one of the red flag phrases above"
}"""


def classify_candidate(name, book_title, book_author, mode, prior_signals, prior_source_url=""):
    cache_prefix = f"classify_v11_{mode}"
    cached = _cache_get(cache_prefix, name, book_title, book_author)
    if cached is not None:
        if not cached.get("source_url") and prior_source_url:
            cached = dict(cached)
            cached["source_url"] = _clean_url(prior_source_url)
        return cached

    default_sig = prior_signals[0] if prior_signals else (
        "blurb" if mode == "blurb" else "social_post"
    )

    prompt = f"""{_CLASSIFY_STATIC}

CANDIDATE TO VERIFY (mode: {mode.upper()}):
Book: "{book_title}" by {book_author}
Endorser: {name}
Prior signal tag(s): {', '.join(prior_signals) if prior_signals else 'unknown'}
Default signal_type if you can't tell: {default_sig}"""
    text = _call_with_search(prompt, max_output_tokens=1500, temperature=0,
                             cache_key="classify")
    result = _extract_json(text)

    def _to_bool(v, default=False):
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower() in ("true", "yes"):
            return True
        if isinstance(v, str) and v.lower() in ("false", "no"):
            return False
        return default

    if not result:
        final = {
            "verified": True,
            "signal_type": default_sig,
            "is_author": False,
            "same_person_confidence": 0.8,
            "source_url": _clean_url(prior_source_url),
            "quote": "",
            "reason": "empty verifier — defaulting to include",
        }
    else:
        verified = result.get("verified")
        if verified is None:
            verified = True
        else:
            verified = _to_bool(verified, True)

        confidence = result.get("same_person_confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.8

        sig = result.get("signal_type") or default_sig
        if sig not in KNOWN_SIGNALS:
            sig = default_sig

        final = {
            "verified": verified,
            "signal_type": sig,
            "is_author": _to_bool(result.get("is_author"), False),
            "same_person_confidence": confidence,
            "source_url": _clean_url(result.get("source_url", "")) or _clean_url(prior_source_url),
            "quote": result.get("quote", "") or "",
            "reason": result.get("reason", "") or "",
        }

    _cache_set(cache_prefix, final, name, book_title, book_author)
    return final


def _should_include(classification, mode):
    if classification.get("is_author"):
        return False, "author of book"
    if classification.get("same_person_confidence", 1.0) < MIN_SAME_PERSON_CONFIDENCE:
        return False, "different person with same name"
    if classification.get("verified") is False:
        return False, classification.get("reason") or "verifier rejected"
    sig = classification.get("signal_type", "")
    if mode == "blurb" and sig in CASUAL_SIGNALS:
        return False, f"signal_type={sig} (casual signal in blurb mode)"
    if mode == "casual" and sig in BLURB_SIGNALS:
        return False, f"signal_type={sig} (blurb signal in casual mode)"
    if mode == "casual" and not classification.get("source_url"):
        return False, "casual share without source URL"
    return True, "ok"


# ============================================================
# RANK
# ============================================================

def rank(verified):
    by_key = {}
    for name, book, c in verified:
        key = _book_dedup_key(book)
        if key not in by_key:
            by_key[key] = {
                "title": book["title"],
                "author": book.get("author", "unknown"),
                "year": book.get("year"),
                "one_line": book.get("one_line", ""),
                "endorsers": [],
                "evidence": [],
                "signal_types": [],
            }
        if len(book.get("title", "")) > len(by_key[key]["title"]):
            by_key[key]["title"] = book["title"]
        if _author_score(book.get("author")) > _author_score(by_key[key]["author"]):
            by_key[key]["author"] = book["author"]
        if name not in by_key[key]["endorsers"]:
            by_key[key]["endorsers"].append(name)
        sig = c.get("signal_type", "unknown")
        if sig not in by_key[key]["signal_types"]:
            by_key[key]["signal_types"].append(sig)

        # Filter out byline-only quotes (e.g. "Bradley Hope, co-author of...")
        quote = c.get("quote", "") or book.get("quote_snippet", "")
        base = re.sub(r"\s*\([^)]*\)", "", name).lower().strip()
        if quote.lower().strip().startswith(base):
            quote = ""

        by_key[key]["evidence"].append({
            "endorser": name,
            "url": c.get("source_url", "") or book.get("source_url", ""),
            "quote": quote,
            "signal_type": sig,
        })

    return sorted(
        by_key.values(),
        key=lambda x: (-len(x["endorsers"]), -(x.get("year") or 0)),
    )


# ============================================================
# ORCHESTRATOR
# ============================================================

def _run_mode(primary, mode, log, force_refresh=False):
    log(f"\n--- {mode.upper()} MODE ---")
    t0 = time.time()

    result = gather_for_mode(primary, mode, log=log, force_refresh=force_refresh)
    raw_candidates = [b for b in result.get("books", []) if b.get("title", "").strip()]

    rejection_reasons = {}
    candidates = []
    for book in raw_candidates:
        if _is_author_of_book(primary, book.get("author", "")):
            log(f"  - '{book['title'][:60]}' (author of book — hard filter)")
            rejection_reasons["author of book (hard filter)"] = (
                rejection_reasons.get("author of book (hard filter)", 0) + 1
            )
            continue
        candidates.append((primary, book))

    log(f"  Classifying {len(candidates)} candidates "
        f"({len(raw_candidates) - len(candidates)} hard-filtered)...")

    verified = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_book = {
            executor.submit(
                classify_candidate,
                primary,
                book["title"],
                book.get("author", ""),
                mode,
                [book.get("signal_type", "unknown")],
                book.get("source_url", ""),
            ): book
            for _, book in candidates
        }
        for future in as_completed(future_to_book):
            book = future_to_book[future]
            try:
                c = future.result()
                include, why = _should_include(c, mode)
                if include:
                    log(f"  + '{book['title'][:60]}' [{c.get('signal_type')}]")
                    verified.append((primary, book, c))
                else:
                    log(f"  - '{book['title'][:60]}' ({why})")
                    rejection_reasons[why] = rejection_reasons.get(why, 0) + 1
            except Exception as e:
                log(f"  error: {e}")
                rejection_reasons["classifier exception"] = (
                    rejection_reasons.get("classifier exception", 0) + 1
                )

    ranked = rank(verified)
    duration = round(time.time() - t0, 2)

    metrics = {
        "mode": mode,
        "duration_s": duration,
        "variants_searched": result.get("variants_searched", []),
        "gather_runs": result.get("gather_runs", 0),
        "candidates_found": len(raw_candidates),
        "hard_filtered": len(raw_candidates) - len(candidates),
        "verified_count": len(verified),
        "rejected_count": sum(rejection_reasons.values()),
        "rejection_reasons": rejection_reasons,
        "final_count": len(ranked),
    }
    log(f"  [metrics] {mode} done in {duration}s: "
        f"{metrics['candidates_found']} candidates -> "
        f"{metrics['final_count']} final "
        f"({metrics['hard_filtered']} hard-filtered)")

    return ranked, metrics


def recommend_from_name(person_name, include_casual=False, log=print, force_refresh=False):
    t_total = time.time()
    log(f"\n[1] Disambiguating '{person_name}'...")
    disambig = disambiguate_person(person_name)
    primary = disambig["primary"]
    log(f"  Resolved to: {primary}")
    if disambig.get("alternatives"):
        log(f"  (Other people with this name: {', '.join(disambig['alternatives'])})")

    blurbs, blurb_metrics = _run_mode(primary, "blurb", log, force_refresh=force_refresh)
    casual, casual_metrics = [], None
    if include_casual:
        casual, casual_metrics = _run_mode(primary, "casual", log, force_refresh=force_refresh)

    total_duration = round(time.time() - t_total, 2)

    telemetry_record = {
        "input_name": person_name,
        "resolved_to": primary,
        "is_ambiguous": disambig.get("is_ambiguous", False),
        "alternatives_count": len(disambig.get("alternatives", [])),
        "include_casual": include_casual,
        "force_refresh": force_refresh,
        "total_duration_s": total_duration,
        "blurb": blurb_metrics,
        "casual": casual_metrics,
        "usage": _usage_snapshot_and_reset(),
    }
    _log_metric(telemetry_record)
    log(f"\n[telemetry] logged to {METRICS_FILE} (total {total_duration}s, "
        f"~${telemetry_record['usage']['est_cost_usd']} API cost, "
        f"{telemetry_record['usage']['web_search_calls']} web searches)")

    return {
        "input_name": person_name,
        "resolved_to": primary,
        "alternatives": disambig.get("alternatives", []),
        "include_casual": include_casual,
        "blurbs": blurbs,
        "casual_shares": casual,
    }


def recommend_from_book(book_title, book_author, log=print):
    raise NotImplementedError(
        f"recommend_from_book is disabled in v10. "
        f"Got call for book='{book_title}' by '{book_author}'. "
        f"Use recommend_from_name(endorser_name) instead."
    )


# ============================================================
# METRICS UTILITIES
# ============================================================

def read_metrics(limit=None):
    if not os.path.exists(METRICS_FILE):
        return []
    records = []
    try:
        with open(METRICS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    records.reverse()
    if limit:
        records = records[:limit]
    return records


def metrics_summary():
    records = read_metrics()
    if not records:
        return "No metrics yet."

    total = len(records)
    avg_total = sum(r.get("total_duration_s", 0) for r in records) / total
    avg_blurb = sum(r.get("blurb", {}).get("duration_s", 0) for r in records) / total
    avg_candidates = sum(r.get("blurb", {}).get("candidates_found", 0) for r in records) / total
    avg_final = sum(r.get("blurb", {}).get("final_count", 0) for r in records) / total
    avg_hard_filtered = sum(r.get("blurb", {}).get("hard_filtered", 0) for r in records) / total
    zero_result = sum(1 for r in records if r.get("blurb", {}).get("final_count", 0) == 0)

    all_rejections = {}
    for r in records:
        for reason, count in r.get("blurb", {}).get("rejection_reasons", {}).items():
            all_rejections[reason] = all_rejections.get(reason, 0) + count
    top_rejections = sorted(all_rejections.items(), key=lambda x: -x[1])[:5]

    lines = [
        f"Metrics summary ({total} queries):",
        f"  Avg total latency: {avg_total:.1f}s",
        f"  Avg blurb latency: {avg_blurb:.1f}s",
        f"  Avg candidates found: {avg_candidates:.1f}",
        f"  Avg hard-filtered: {avg_hard_filtered:.1f}",
        f"  Avg final results: {avg_final:.1f}",
        f"  Zero-result queries: {zero_result} ({100*zero_result/total:.0f}%)",
        f"  Top rejection reasons:",
    ]
    for reason, count in top_rejections:
        lines.append(f"    {count}x  {reason}")
    return "\n".join(lines)


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(text):
    if not text:
        return {}
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    cleaned = candidate
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    cleaned = re.sub(
        r':\s*(?!true\b|false\b|null\b)([A-Za-z_][A-Za-z0-9_/.\-]*)\s*([,}\]])',
        r': null\2',
        cleaned,
    )
    cleaned = re.sub(r':\s*\d+\?+\s*([,}\]])', r': null\1', cleaned)
    cleaned = re.sub(r':\s*\([^()"]*\)\s*([,}\]])', r': null\1', cleaned)
    cleaned = re.sub(r':\s*\([^"{}\[\]]*?(,(?=\s*")|[}\]])', r': null\1', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        books_match = re.search(r'"books"\s*:\s*\[', cleaned)
        if books_match:
            books_start = books_match.end()
            depth = 0
            last_complete = books_start
            in_string = False
            escape = False
            for i in range(books_start, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        last_complete = i + 1
            repaired = cleaned[:last_complete] + "]}"
            return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    for end_pos in range(end, start, -1):
        sub = text[start:end_pos + 1]
        if sub.endswith("}"):
            try:
                return json.loads(sub)
            except json.JSONDecodeError:
                continue

    repaired_text = _call_plain(
        "Convert the following text into VALID JSON. If it describes book "
        "endorsements, use the schema "
        '{"endorser": str, "books": [{"title": str, "author": str, '
        '"year": int or null, "one_line": str, "signal_type": str, '
        '"quote_snippet": str, "source_url": str, "notes": str}]}. '
        "Use null for unknown fields. If no books are mentioned, return "
        '{"books": []}. Return ONLY the JSON, no markdown.\n\n' + text[:8000],
        max_output_tokens=4000,
    )
    if repaired_text:
        rs, re_ = repaired_text.find("{"), repaired_text.rfind("}")
        if rs != -1 and re_ != -1:
            try:
                fixed = re.sub(r',\s*([}\]])', r'\1', repaired_text[rs:re_ + 1])
                result = json.loads(fixed)
                print("  [info] JSON recovered via LLM repair pass")
                return result
            except json.JSONDecodeError:
                pass
    print(f"  [warning] failed to parse JSON; first 200 chars: {text[:200]}")
    return {}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    if "--metrics" in sys.argv:
        print(metrics_summary())
        sys.exit(0)

    if "--test-author-filter" in sys.argv:
        test_cases = [
            ("Bradley Hope (journalist)", "Tom Wright and Bradley Hope", True),
            ("Bradley Hope (journalist)", "Justin Scheck (and Bradley P. Hope as co-author)", True),
            ("Bradley Hope (journalist)", "Faiz Siddiqui", False),
            ("Bradley Hope", "Bradley Hope", True),
            ("Stephen King", "Owen King", False),
            ("Stephen King", "Stephen King", True),
            ("Ken Griffin (Citadel founder)", "Kenneth C. Griffin", True),
            ("Bono", "Bono", False),
            ("Yo-Yo Ma (cellist)", "Yo-Yo Ma", True),
            ("J.K. Rowling", "Robert Galbraith", False),
        ]
        print("Testing _is_author_of_book:")
        all_pass = True
        for endorser, author, expected in test_cases:
            got = _is_author_of_book(endorser, author)
            mark = "✓" if got == expected else "✗"
            if got != expected:
                all_pass = False
            print(f"  {mark}  endorser='{endorser}' author='{author}' "
                  f"-> got {got}, expected {expected}")
        print(f"\n{'All tests passed' if all_pass else 'SOME TESTS FAILED'}")
        sys.exit(0 if all_pass else 1)

    force_refresh = "--force-refresh" in sys.argv
    include_casual = "--casual" in sys.argv
    args = [a for a in sys.argv[1:]
            if a not in ("--casual", "--force-refresh")]
    name = args[0] if args else "Bradley Hope"

    result = recommend_from_name(
        name,
        include_casual=include_casual,
        force_refresh=force_refresh,
    )

    print("\n" + "=" * 60)
    print(f"BLURBS by {result['resolved_to']}")
    print("=" * 60)
    blurbs = result.get("blurbs", [])
    if not blurbs:
        print(f"\n{result['resolved_to']} has not written blurbs for any books, yet.")
    else:
        for i, rec in enumerate(blurbs, 1):
            year = rec.get("year") or "n/a"
            sig = ", ".join(rec.get("signal_types", []))
            print(f"\n{i}. {rec['title']} ({year})")
            print(f"   by {rec['author']}")
            print(f"   type: {sig}")
            urls = [ev["url"] for ev in rec.get("evidence", []) if ev.get("url")]
            if urls:
                print(f"   source: {urls[0]}")

    if include_casual:
        print("\n" + "=" * 60)
        print(f"CASUAL SHARES by {result['resolved_to']}")
        print("=" * 60)
        casual = result.get("casual_shares", [])
        if not casual:
            print(f"\nNo casual shares found for {result['resolved_to']}.")
        else:
            for i, rec in enumerate(casual, 1):
                year = rec.get("year") or "n/a"
                sig = ", ".join(rec.get("signal_types", []))
                print(f"\n{i}. {rec['title']} ({year})")
                print(f"   by {rec['author']}")
                print(f"   type: {sig}")
                urls = [ev["url"] for ev in rec.get("evidence", []) if ev.get("url")]
                if urls:
                    print(f"   source: {urls[0]}")

    with open("recommendations.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to recommendations.json")