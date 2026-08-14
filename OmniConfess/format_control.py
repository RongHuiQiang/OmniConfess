# -*- coding: utf-8 -*-
"""Format control: turn the output format from something hoped for into a
decoding-time constraint.

Why this exists
---------------
The judgement branch's stage-1 prefix ends with "Visual Checklist:\n- " and then
leaves the model to produce the rest of the structure on its own. Measured over a
full run, that fails in several systematic ways: almost every generation opens with
a list marker, because the prefix ends on a dangling "- "; a noticeable fraction
copies the bracketed placeholder out of the system prompt verbatim; about half the
samples never write the second (auditory) checklist at all, including half of the
questions that ask about sound; and essentially nothing starts with "Description:",
which the v0 contract demands, because the prefix and the system prompt contradict
each other. Stop strings were also inert, since safe_model_generate dropped them
along with the tokenizer.

Note that the missing second checklist is not a budget problem, so raising
max_new_tokens does not help: the samples missing it are on average shorter than
those that have it and hit the cap less often. The model simply stops early.

The conclusion is that prompt wording cannot fix this. What determines the format
is who emits the structural tokens.

Three tools
-----------
1. TemplateForcingLogitsProcessor -- skeleton forcing. The output is split into
   alternating literals and slots: literals such as "Auditory Checklist:\n- " are
   emitted token by token by us, and the model only generates inside the slots.
   Format compliance is guaranteed by construction, and it costs no extra forward
   pass, being just a logits processor on the same two generate calls.
2. PromptAntiCopyLogitsProcessor -- anti-copy. Bans any token that would continue
   the generated tail into an n-gram that already appears in the prompt. n = 6 by
   default: six consecutive identical tokens are essentially never a coincidence,
   whereas legitimate repetition of a content word from the question is short
   enough not to be caught. It is generic and does not need to know what the
   placeholders look like.
3. build_stopping_criteria -- restores the dropped stop_strings using Hugging
   Face's StopStringCriteria, which under inputs_embeds generation sees only the
   generated part, exactly as wanted.

Design constraints
------------------
- Everything is off by default. Without a logits_processor argument, behaviour is
  unchanged and previously produced runs stay byte-reproducible.
- The processors are single-use: build a new one per generate call. The internal
  cursor guards itself with a reset, so if input_ids gets shorter, meaning the same
  object was reused for a new round, it resets automatically.
- Only batch size 1 with greedy decoding is supported, which is how the judgement
  branch calls it. A larger batch raises rather than silently misbehaving.

Self-check (no GPU and no model weights needed, only a tokenizer):
    python format_control.py
"""
from collections import defaultdict

import torch
from transformers import (LogitsProcessor, LogitsProcessorList,
                          StoppingCriteria, StoppingCriteriaList)

NEG_INF = -float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
def lit(text):
    """A literal span: this text is emitted token by token by us, with no room for the model to interject."""
    return {"kind": "lit", "text": text}


def slot(name, min_tokens=4, max_tokens=64, ban_first=(), allow=None, close=True,
         desc=None):
    """Free slot: the model generates freely here.

    min_tokens  below this many tokens, the terminator is banned (guards against an
                empty slot)
    max_tokens  at this many tokens the terminator is forced (guards against
                runaway generation crowding out later spans)
    ban_first   tokens banned as the slot's first token, used to suppress list
                markers such as "1", "1." or "-"
    allow       if given, the whole slot may only draw from this set; the verdict
                slot uses it to lock the output to Yes/No
    close       whether to append a terminator (newline) once the slot is full. The
                final slot must use close=False, otherwise one extra token is
                generated. The second-stage verdict slot is especially sensitive to
                this: a forced extra token has a log_softmax of exactly 0, which
                halves the mean of all_logprobs.
    desc        a plain-language description of what belongs in this slot. It is
                used only by render_contract() to generate the format section of
                the prompt, so that the contract and the skeleton come from one
                definition.
    """
    return {"kind": "slot", "name": name, "min": min_tokens, "max": max_tokens,
            "ban_first": set(ban_first), "close": bool(close),
            "desc": desc or name,
            "allow": None if allow is None else set(allow)}


def compile_segments(segments, tokenizer, terminator="\n"):
    """Compile lit/slot descriptions into spans carrying token ids. The terminator defaults to a newline."""
    term_ids = _ids_of(tokenizer, terminator)
    assert term_ids, f"terminator {terminator!r} encodes to no tokens"
    term_id = term_ids[0]
    out = []
    for seg in segments:
        s = dict(seg)
        if s["kind"] == "lit":
            s["ids"] = tokenizer.encode(s["text"], add_special_tokens=False)
            assert s["ids"], f"literal {s['text']!r} encodes to an empty token sequence"
        else:
            s["stop"] = set(_newline_ids(tokenizer))
            s["term"] = term_id
        out.append(s)
    return out


def _ids_of(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _newline_ids(tokenizer):
    """All pure-whitespace tokens that begin with a newline; a slot treats one as the end of its line."""
    got = set()
    for t in ("\n", "\n\n", " \n"):
        ids = _ids_of(tokenizer, t)
        if len(ids) == 1:
            got.add(ids[0])
    return got or {_ids_of(tokenizer, "\n")[0]}


class _Cursor:
    """A cursor over the span sequence. Advances token by token and answers what must be forced next."""

    def __init__(self, segs):
        self.segs = segs
        self.reset()

    def reset(self):
        self.si = 0
        self.k = 0
        self.force_term = False

    def action(self):
        """('done',) | ('force', tok_id) | ('free', seg, k)"""
        if self.si >= len(self.segs):
            return ("done",)
        seg = self.segs[self.si]
        if seg["kind"] == "lit":
            return ("force", seg["ids"][self.k])
        if self.force_term:
            return ("force", seg["term"])
        return ("free", seg, self.k)

    def update(self, tok):
        if self.si >= len(self.segs):
            return
        seg = self.segs[self.si]
        if seg["kind"] == "lit":
            self.k += 1
            if self.k >= len(seg["ids"]):
                self.si += 1
                self.k = 0
            return
        self.k += 1
        if self.force_term:
            self.force_term = False
            self.si += 1
            self.k = 0
            return
        if self.k >= seg["min"] and tok in seg["stop"]:
            self.si += 1
            self.k = 0
            return
        if self.k >= seg["max"]:
            if seg.get("close", True):
                self.force_term = True
            else:
                self.si += 1
                self.k = 0


class TemplateForcingLogitsProcessor(LogitsProcessor):
    """Skeleton forcing. Every structural token is emitted by this processor and the
    model only fills the slots.

    `stopped` is set once the whole template has been walked, which
    TemplateFinishedCriteria uses to end generation.
    """

    def __init__(self, segs, eos_id, extra_ban=()):
        self.cur = _Cursor(segs)
        self.eos_id = eos_id
        self.extra_ban = set(extra_ban)
        self._n_seen = 0
        self.trace = []

    def _sync(self, gen):
        if len(gen) < self._n_seen:
            self.cur.reset()
            self._n_seen = 0
            self.trace.clear()
        for tok in gen[self._n_seen:]:
            self.cur.update(tok)
        self._n_seen = len(gen)

    @property
    def finished(self):
        return self.cur.si >= len(self.cur.segs)

    def __call__(self, input_ids, scores):
        assert input_ids.shape[0] == 1, "format control only supports batch size 1 with greedy decoding"
        self._sync(input_ids[0].tolist())
        act = self.cur.action()
        if act[0] == "done":
            self.trace.append(("done", ""))
            out = torch.full_like(scores, NEG_INF)
            out[:, self.eos_id] = 0.0
            return out
        if act[0] == "force":
            self.trace.append(("force", act[1]))
            out = torch.full_like(scores, NEG_INF)
            out[:, act[1]] = 0.0
            return out
        _, seg, k = act
        self.trace.append(("free", seg["name"]))
        if seg["allow"] is not None:
            mask = torch.full_like(scores, NEG_INF)
            idx = list(seg["allow"])
            mask[:, idx] = scores[:, idx]
            scores = mask
        else:
            ban = set(self.extra_ban)
            if k == 0:
                ban |= seg["ban_first"]
            if k < seg["min"]:
                ban |= seg["stop"]
            ban.add(self.eos_id)
            if ban:
                scores[:, list(ban)] = NEG_INF
        return scores


class TemplateFinishedCriteria(StoppingCriteria):
    """Stop as soon as the template is complete, rather than waiting for
    max_new_tokens.

    Calling _sync here is necessary rather than redundant. Hugging Face's loop runs
    logits processor, then sampling, then the stopping criteria, so the processor
    only ever sees the pre-sampling sequence and its `finished` flag lags by one
    token. Reusing that flag would take one extra step after the template's last
    token lands, and on that step the processor forces eos, producing an extra
    token whose log_softmax is exactly 0 and which dilutes the mean of
    all_logprobs. _sync is idempotent, tracking how many tokens it has seen, so a
    later call from the processor does not advance it twice.
    """

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, input_ids, scores, **kw):
        self.proc._sync(input_ids[0].tolist())
        return torch.tensor([self.proc.finished] * input_ids.shape[0],
                            device=input_ids.device, dtype=torch.bool)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
class PromptAntiCopyLogitsProcessor(LogitsProcessor):
    """Ban tokens that would continue the generated tail into an n-gram already
    present in the prompt.

    This is a generic echo gate and does not need to know what the placeholders
    look like. The choice of n = 6 is a trade-off: six consecutive identical tokens
    are essentially never a coincidence, whereas legitimate repetition of a content
    word, such as the object being asked about, is usually only one to three tokens
    and is not caught.
    """

    def __init__(self, prompt_ids, n=6, skip_ids=()):
        self.n = int(n)
        assert self.n >= 2
        self.map = defaultdict(set)
        skip = set(skip_ids)
        ids = [int(t) for t in prompt_ids if int(t) not in skip]
        for i in range(len(ids) - self.n + 1):
            self.map[tuple(ids[i:i + self.n - 1])].add(ids[i + self.n - 1])
        self.blocked = 0

    def __call__(self, input_ids, scores):
        gen = input_ids[0].tolist()
        if len(gen) < self.n - 1:
            return scores
        bad = self.map.get(tuple(gen[-(self.n - 1):]))
        if bad:
            idx = list(bad)
            if len(idx) < scores.shape[-1]:
                scores[:, idx] = NEG_INF
                self.blocked += len(idx)
        return scores


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
def build_stopping_criteria(tokenizer, stop_strings, template_proc=None):
    """safe_model_generate strips stop_strings along with the tokenizer, which makes
    every stop string in the project inert. This restores them with Hugging Face's
    StopStringCriteria."""
    crits = []
    if stop_strings:
        from transformers import StopStringCriteria
        crits.append(StopStringCriteria(tokenizer=tokenizer,
                                        stop_strings=list(stop_strings)))
    if template_proc is not None:
        crits.append(TemplateFinishedCriteria(template_proc))
    return StoppingCriteriaList(crits) if crits else None


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
def list_marker_ids(tokenizer):
    """List marker tokens. Almost every generation opens with one, because the prefix ends on a dangling "- "."""
    got = set()
    for s in ["1", "2", "3", "4", "5", "1.", "2.", "3.", "-", "*", "•",
              " 1", " 2", " -", "1)", "(1", "#"]:
        ids = _ids_of(tokenizer, s)
        if len(ids) == 1:
            got.add(ids[0])
    return got


def verdict_ids(tokenizer):
    """The tokens allowed immediately after `\\boxed{`, which are only the Yes and No
    families.

    Locking this also removes the need for the one-sided parsing fallback that
    defaulted an unparsable verdict to "No": the verdict can no longer fail to parse.
    """
    got = set()
    for s in ["Yes", "No", "yes", "no", "YES", "NO"]:
        ids = _ids_of(tokenizer, s)
        if ids:
            got.add(ids[0])
    return got


_CHECKLIST_HEAD = {"visual": "Visual Checklist:\n- ",
                   "auditory": "Auditory Checklist:\n- "}
_SLOT_DESC = {
    "visual": "objective facts visible in the video, no inference",
    "auditory": "sounds actually present in the audio; say so if a visible "
                "object's sound is absent",
    "queried": "the exact item the question asks about, and which stream it "
               "was found in",
}


def cmm_stage1_segments(tokenizer, queried_line=True, first="visual"):
    """The cmm stage-1 skeleton. The prefix has already emitted the header of the
    `first` checklist, so the first span is a slot.

    Budget: 64 tokens for the visual checklist, 64 for the auditory one and 28 for
    the item asked about, plus 12 tokens of literals, for a worst case of 168 and a
    cap of 200. That is cheaper than the status quo, and because the second
    checklist's header is emitted by us, "never wrote the auditory checklist" drops
    to zero by construction. That failure was never a budget problem: the samples
    missing the section were shorter than those that had it, so the model was
    stopping early of its own accord.

    The literals carry no leading newline, because a slot always ends on a newline
    already, whether the model emits one itself or we append one on hitting the
    cap; adding another would produce a blank line.

    Why `first` is configurable
    ---------------------------
    Which checklist comes first is not an aesthetic question: whichever is written
    last sits closest to the verdict token. In cmm the distractor is determined by
    the question itself, with no exceptions in the dataset: "Did you see ..."
    questions are trapped by over-relying on audio, so the deciding evidence is
    visual, and "Did you hear ..." questions are trapped by over-relying on vision,
    so the deciding evidence is auditory. A fixed first="visual" therefore always
    puts the auditory checklist closest to the verdict, which for every "see"
    question places the distracting modality in the most influential position.
    `first` should therefore be the modality NOT being asked about, so the one being
    asked about lands last.

    `first` must match the prefix the caller actually uses, or the skeleton is
    misaligned by a whole span, with the prefix writing "Visual Checklist:" and the
    skeleton emitting it again. Do not guess it: read it out of the prefix with
    first_from_prefix() and assert it with assert_prefix_matches(). The default of
    "visual" matches the existing prefix, so omitting the argument leaves behaviour
    byte-identical.
    """
    assert first in _CHECKLIST_HEAD, f"first must be visual or auditory, got {first!r}"
    other = "auditory" if first == "visual" else "visual"
    ban = list_marker_ids(tokenizer)
    segs = [
        slot(first, min_tokens=5, max_tokens=64, ban_first=ban,
             desc=_SLOT_DESC[first]),
        lit(_CHECKLIST_HEAD[other]),
        slot(other, min_tokens=5, max_tokens=64, ban_first=ban,
             close=queried_line, desc=_SLOT_DESC[other]),
    ]
    if queried_line:
        segs += [lit("Queried item: "),
                 slot("queried", min_tokens=4, max_tokens=28, ban_first=ban,
                      close=False, desc=_SLOT_DESC["queried"])]
    return compile_segments(segs, tokenizer)


def first_from_prefix(prefix):
    """Read which checklist the stage-1 prefix has already opened, giving the
    skeleton's `first`.

    Reading rather than guessing matters because the prefix has several possible
    sources, and if any of them changes its wording the skeleton must follow.
    Returns None when it cannot be determined, in which case the caller should not
    enable skeleton forcing.
    """
    tail = (prefix or "").rstrip()
    for name, head in _CHECKLIST_HEAD.items():
        if tail.endswith(head.rstrip()):
            return name
    return None


def assert_prefix_matches(segs, prefix):
    """Assert that skeleton and prefix line up exactly: the checklist the prefix has
    opened must not be emitted again by the skeleton.

    This mechanises the "change the prefix and you must change this too" rule. The
    symptom of a misalignment is two "Visual Checklist:" headers in the output,
    which is very hard to notice by eye in a log.
    """
    if not segs or segs[0]["kind"] != "slot":
        raise AssertionError("the skeleton must start with a slot, since the prefix already emitted the header")
    opened = first_from_prefix(prefix)
    if opened is None:
        return
    lits = [s["text"] for s in segs if s["kind"] == "lit"]
    dup = _CHECKLIST_HEAD[opened]
    if any(dup.rstrip() in t for t in lits):
        raise AssertionError(
            f"the prefix already opened the {opened} checklist and the skeleton emitted {dup!r} again, so it is misaligned by a whole span")
    if segs[0]["name"] != opened:
        raise AssertionError(
            f"the prefix opened the {opened} checklist but the skeleton's first slot is named {segs[0]['name']!r}")


def _head_line(text):
    """Render a literal as a header line of the contract.

    Header literals all end on a dangling "- ", which is the hook that invites the
    model to continue with bullet points and is also why so many outputs open with
    a list marker. That suffix must be removed when rendering the contract,
    otherwise the header and the slot line that follows each emit a "-" and the
    contract gains a stray bare "-" line.
    """
    return text.rstrip().rstrip("-").rstrip()


def render_contract(segs, prefix="", verdict_line=r"Judgment: \boxed{Yes} or \boxed{No}"):
    """Render the skeleton into the format section of the prompt, so that the
    contract and the forcing come from one definition.

    The motivation is that the system prompt and the scaffold currently state two
    different formats, which the model cannot possibly satisfy at once: v0's strict
    output format asks for "Description:" followed by one sentence, while the
    scaffold pushes a "Visual Checklist:" bullet list. In practice almost nothing
    follows either, a noticeable fraction copies the prompt text into the answer,
    and nothing at all begins with "Description:". That is not disobedience; it is
    our own contract contradicting itself.

    The format section of the prompt should therefore be generated by this function
    rather than written by hand. Handwriting drifts; generating it means changing
    the skeleton is changing the contract, and the two can no longer diverge.
    """
    lines = []
    opened = first_from_prefix(prefix)
    if opened:
        lines.append(_head_line(_CHECKLIST_HEAD[opened]))
    for s in segs:
        if s["kind"] == "lit":
            lines.append(_head_line(s["text"]))
        else:
            lines.append(f"- <{s['min']}-{s['max']} tokens: {s['desc']}>"
                         if s["name"] != "queried" else
                         f"<{s['min']}-{s['max']} tokens: {s['desc']}>")
    if verdict_line:
        lines.append(verdict_line)
    out = []
    for ln in lines:
        if out and out[-1].endswith(":") and ln.startswith("<"):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return "\n".join(out)



def phd_stage1_segments(tokenizer, queried_line=True):
    """The phd stage-1 skeleton. phd is image-only with no audio track, so it must
    have no auditory checklist.

    Forcing a checklist for an input the model does not have amounts to instructing
    it to invent one, which is exactly the failure this project is trying to fix.
    """
    ban = list_marker_ids(tokenizer)
    segs = [slot("visual", min_tokens=5, max_tokens=96, ban_first=ban,
                 close=queried_line, desc=_SLOT_DESC["visual"])]
    if queried_line:
        segs += [lit("Queried item: "),
                 slot("queried", min_tokens=4, max_tokens=28, ban_first=ban,
                      close=False, desc=_SLOT_DESC["queried"])]
    return compile_segments(segs, tokenizer)


def cmm_stage2_segments(tokenizer):
    """The stage-2 skeleton. The prefix already supplies "Judgment: \\boxed{", so only
    a single locked verdict slot remains.

    close=False makes it generate exactly one token and stop. That is a hard
    requirement for stage 2; see the note on close in slot().
    """
    return compile_segments(
        [slot("verdict", min_tokens=1, max_tokens=1, close=False,
              allow=verdict_ids(tokenizer))],
        tokenizer)


def make_processors(tokenizer, segs=None, prompt_ids=None, anti_copy_n=6):
    """Assemble the LogitsProcessorList. Either tool can be enabled alone, which makes a 2x2 factorial easy."""
    procs, tf = [], None
    if prompt_ids is not None and anti_copy_n:
        procs.append(PromptAntiCopyLogitsProcessor(prompt_ids, n=anti_copy_n))
    if segs:
        tf = TemplateForcingLogitsProcessor(segs, eos_id=tokenizer.eos_token_id)
        procs.append(tf)
    return (LogitsProcessorList(procs) if procs else None), tf


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    MP = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Omni-7B"
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    V = len(tok)
    print(f"tokenizer: {MP}  vocab {V}  eos={tok.eos_token_id}")

    segs = cmm_stage1_segments(tok)
    print(f"\nstage-1 spans: {len(segs)}")
    for s in segs:
        print(f"   {s['kind']:<5} " + (repr(s['text']) if s['kind'] == 'lit'
                                       else f"{s['name']}  min={s['min']} max={s['max']}"))

    # -- self-checks: configurable order, prefix readback, contract rendering --
    print("\n" + "=" * 78)
    print(" configurable order + prefix alignment assertions")
    print("=" * 78)
    _VP, _AP = "…\nVisual Checklist:\n- ", "…\nAuditory Checklist:\n- "
    assert first_from_prefix(_VP) == "visual"
    assert first_from_prefix(_AP) == "auditory"
    assert first_from_prefix("…\nDescription: ") is None
    print("  ok: first_from_prefix reads all three prefixes correctly (None when unrecognised)")

    _d = cmm_stage1_segments(tok)
    _v = cmm_stage1_segments(tok, first="visual")
    assert [x.get("ids") or x["name"] for x in _d] == \
           [x.get("ids") or x["name"] for x in _v], "the default no longer equals first='visual'"
    assert _d[0]["name"] == "visual" and "Auditory" in _d[1]["text"]
    print("  ok: omitting first equals first='visual', matching the existing behaviour byte for byte")

    _a = cmm_stage1_segments(tok, first="auditory")
    assert _a[0]["name"] == "auditory" and "Visual Checklist" in _a[1]["text"]
    print("  ok: with first='auditory' the auditory slot comes first and we emit the Visual header")
    print(f"     => 'see' questions (which over-rely on audio) should use first='auditory', "
          f"so the visual checklist lands next to the verdict")

    assert_prefix_matches(_v, _VP)
    assert_prefix_matches(_a, _AP)
    for bad_segs, bad_pre, why in ((_v, _AP, "skeleton starts visual but the prefix opened auditory"),
                                   (_a, _VP, "skeleton starts auditory but the prefix opened visual")):
        try:
            assert_prefix_matches(bad_segs, bad_pre)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"misalignment was not caught: {why}")
    print("  ok: passes when aligned and raises AssertionError when misaligned (both directions tested)")

    print("\n" + "=" * 78)
    print(" contract rendering (the prompt format section is generated from the skeleton, not handwritten)")
    print("=" * 78)
    for pre, lbl in ((_VP, "hear question: visual first, auditory next to the verdict"),
                     (_AP, "see question: auditory first, visual next to the verdict")):
        f = first_from_prefix(pre)
        print(f"\n  ── {lbl} ──")
        print("  " + render_contract(cmm_stage1_segments(tok, first=f),
                                     prefix=pre).replace("\n", "\n  "))
    print("\n  -- phd (image only, no auditory checklist) --")
    print("  " + render_contract(phd_stage1_segments(tok),
                                 prefix=_VP).replace("\n", "\n  "))

    def run(policy, n_max=400, prompt_ids=None, segments=None):
        """Run the whole state machine against a fake model, recording every token.

        The loop order copies Hugging Face exactly: logits processor, then sampling,
        then the stopping criteria. Copying it matters, because the fact that the
        processor's `finished` flag lags by one token is only exposed by running in
        the real order.
        """
        sg = segs if segments is None else segments
        lp, tf = make_processors(tok, sg, prompt_ids=prompt_ids)
        crit = build_stopping_criteria(tok, None, template_proc=tf)
        gen = []
        for _ in range(n_max):
            sc = lp(torch.tensor([gen], dtype=torch.long), torch.zeros(1, V))
            allowed = torch.nonzero(sc[0] > NEG_INF).flatten().tolist()
            gen.append(policy(len(gen), allowed))
            if bool(crit(torch.tensor([gen], dtype=torch.long), None)[0]):
                break
        return gen, tf

    print("\n" + "═" * 76)
    print(" test 1: a parrot model that tries to copy the prompt placeholder at every step; the skeleton still holds")
    print("═" * 76)
    echo = tok.encode("One flowing sentence describing the scene and the EXACT sounds "
                      "heard, like: 'In a room, a man spoke by a window where birds "
                      "were seen but not heard.'", add_special_tokens=False)
    st = {"i": 0}

    def echo_policy(k, allowed):
        for _ in range(len(echo)):
            t = echo[st["i"] % len(echo)]
            st["i"] += 1
            if t in allowed:
                return t
        return allowed[0]

    gen, tf = run(echo_policy)
    text = tok.decode(gen)
    print(f"generated {len(gen)} tokens, template complete={tf.finished}")
    print("-- full output with the prefix prepended --")
    print("Visual Checklist:\n- " + text)
    ok = ("Auditory Checklist:" in text) and ("Queried item:" in text)
    print(f"\nok: both checklists and the Queried line are present: {ok}")
    assert ok

    print("\n" + "═" * 76)
    print(" test 2: a model insisting on list markers and newlines; ban_first and min take effect")
    print("═" * 76)
    marks = sorted(list_marker_ids(tok))
    nl = sorted(_newline_ids(tok))
    print(f"{len(marks)} list-marker tokens: {[tok.decode([i]) for i in marks]}")
    print(f"{len(nl)} newline tokens: {[repr(tok.decode([i])) for i in nl]}")

    def spam_policy(k, allowed):
        for t in marks + nl:
            if t in allowed:
                return t
        return allowed[0]

    gen, tf = run(spam_policy)
    text = tok.decode(gen)
    print(f"generated {len(gen)} tokens, template complete={tf.finished}")
    print("-- output (first 200 chars) --")
    print(repr(("Visual Checklist:\n- " + text)[:200]))
    first = tok.decode([gen[0]])
    print(f"\nok: the first token is not a list marker: {gen[0] not in list_marker_ids(tok)} (it is {first!r})")
    assert gen[0] not in list_marker_ids(tok)

    print("\n" + "═" * 76)
    print(" test 3: anti-copy, a 6-gram from the prompt cannot be continued")
    print("═" * 76)
    pid = tok.encode("STRICT OUTPUT FORMAT:\nDescription: [One flowing sentence "
                     "describing the scene and the EXACT sounds heard, like: 'In a "
                     "room, a man spoke by a window where birds were seen but not "
                     "heard.']", add_special_tokens=False)
    st["i"] = 0
    gen_a, _ = run(echo_policy)
    st["i"] = 0
    gen_b, _ = run(echo_policy, prompt_ids=pid)
    ta, tb = tok.decode(gen_a), tok.decode(gen_b)

    def longest_common_run(a, b_ids):
        """Length of the longest run of tokens in a (the generation) that also appears contiguously in b_ids (the prompt)."""
        bs = {tuple(b_ids[i:i + L]) for L in range(1, 40)
              for i in range(len(b_ids) - L + 1)}
        best = 0
        for L in range(1, min(40, len(a)) + 1):
            if any(tuple(a[i:i + L]) in bs for i in range(len(a) - L + 1)):
                best = L
        return best

    la, lb = longest_common_run(gen_a, pid), longest_common_run(gen_b, pid)
    print(f"anti-copy off: longest contiguous overlap with the prompt is {la} tokens")
    print(f"anti-copy on: longest contiguous overlap with the prompt is {lb} tokens (n=6 implies an upper bound of 5)")
    print(f"\noutput with anti-copy off: {ta[:110]!r}")
    print(f"output with anti-copy on: {tb[:110]!r}")
    assert lb <= 5, f"anti-copy failed to hold: {lb} > 5"
    print(f"\nok: copied run reduced to at most 5 tokens (was {la})")

    print("\n" + "═" * 76)
    print(" test 4: the stage-2 verdict slot is locked to Yes/No, so parsing cannot fail, and exactly one token is generated")
    print("═" * 76)
    s2 = cmm_stage2_segments(tok)
    allow = sorted(verdict_ids(tok))
    print(f"allowed verdict-initial tokens: {[(i, tok.decode([i])) for i in allow]}")
    lp2, tf2 = make_processors(tok, s2)
    sc = lp2(torch.tensor([[]], dtype=torch.long), torch.zeros(1, V))
    n_ok = int((sc[0] > NEG_INF).sum())
    print(f"candidate tokens at the first step: {n_ok} (should equal {len(allow)})")
    assert n_ok == len(allow)

    def yes_policy(k, allowed):
        return allowed[0]

    g2, tf2b = run(yes_policy, segments=s2)
    print(f"generated {len(g2)} tokens: {[tok.decode([t]) for t in g2]}  template complete={tf2b.finished}")
    assert len(g2) == 1, f"the verdict slot generated {len(g2)} tokens, which would dilute the mean logprob"
    print("\nok: the verdict can only be Yes or No and is exactly one token, so "
          "the one-sided fallback to \"No\" is unreachable and the logprob convention is unchanged")

    print("\n" + "═" * 76)
    print(" test 5: the phd skeleton has no auditory checklist (phd is image only, so forcing an audio section would be instructing the model to invent one)")
    print("═" * 76)
    ps = phd_stage1_segments(tok)
    lits = " ".join(s["text"] for s in ps if s["kind"] == "lit")
    print(f"phd spans: {len(ps)}, literals: {lits!r}")
    assert "Auditory" not in lits, "the phd skeleton must not contain an auditory checklist"
    st["i"] = 0
    g5, tf5 = run(echo_policy, segments=ps)
    t5 = tok.decode(g5)
    print(f"generated {len(g5)} tokens, template complete={tf5.finished}")
    print("-- full output with the prefix prepended (first 240 chars) --")
    print(("Visual Checklist:\n- " + t5)[:240])
    assert "Auditory Checklist:" not in t5 and "Queried item:" in t5
    print("\nok: phd has only the visual checklist and the Queried line")

    print("\n" + "═" * 76)
    print(" all self-checks passed")
    print("═" * 76)
