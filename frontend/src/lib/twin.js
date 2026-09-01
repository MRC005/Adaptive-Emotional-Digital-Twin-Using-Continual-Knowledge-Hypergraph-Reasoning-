/**
 * LAYER 1 in the browser -- perception, events, hypergraph, twin.
 *
 * This mirrors the Python modules it is named after, field for field:
 *   aedt/emotion/context.py      -> extractContext
 *   aedt/emotion/events.py       -> buildEvent, FieldSource
 *   aedt/hypergraph/event_graph.py -> buildEventHypergraph
 *   aedt/twin/personal_twin.py   -> PersonalTwin
 *
 * The one thing it CANNOT mirror is the Transformer. RoBERTa does not run in a
 * browser, so emotion detection here is either
 *   - the backend, when one is configured (label: the real model), or
 *   - a lexicon baseline, labelled as a baseline everywhere it appears.
 * The interface shows which one produced each label. That is the whole reason
 * `backend` travels with every prediction.
 *
 * PRIVACY. Everything below stays in this tab. History is held in memory and,
 * if the viewer allows it, in localStorage on their own device. Nothing is
 * sent anywhere except the single check-in sentence, and only when the user
 * has a backend configured and has opted into using it.
 */

export const FieldSource = {
  EXTRACTED: "extracted", USER_REPORTED: "user_reported", MODEL: "model",
  INFERRED: "inferred", CORRECTED: "corrected", UNKNOWN: "unknown",
};

export const CONTEXT_FIELDS = ["emotion", "statedEmotion", "event", "time_context", "sleep",
                               "activity", "social", "workload", "location"];

export const FIELD_LABELS = {
  emotion: "Feeling",
  statedEmotion: "You said", event: "What happened", time_context: "When",
  sleep: "Sleep", activity: "Activity", social: "People",
  workload: "How busy", location: "Where",
};

/** Options offered when the user corrects a field. */
export const FIELD_OPTIONS = {
  emotion: ["anxiety", "stress", "sadness", "anger", "joy", "gratitude",
            "calm", "confusion", "neutral"],
  event: ["examination", "deadline", "hospital appointment", "presentation",
          "work", "study", "travel", "family", "social event", "conflict",
          "bereavement", "finances"],
  time_context: ["today", "tomorrow", "yesterday", "last night", "tonight",
                 "this week", "next week"],
  sleep: ["poor", "good"],
  activity: ["low", "high"],
  social: ["isolated", "with others"],
  workload: ["low", "high"],
  location: ["home", "work", "campus", "hospital"],
};

const P = (value, source, confidence = null, evidence = "") =>
  ({ value, source, confidence, evidence });
const UNKNOWN = () => P(null, FieldSource.UNKNOWN);
const known = (p) => Boolean(p && p.source !== FieldSource.UNKNOWN && p.value != null);

/* ---------------------------------------------------- context extraction */
/* Kept identical to aedt/emotion/context.py. When one changes the other must. */
const EVENT_LEXICON = {
  "hospital appointment": ["hospital", "doctor", "appointment", "clinic", "surgery",
                           "scan", "x-ray", "checkup", "check-up", "medical"],
  examination: ["exam", "examination", "finals", "midterm", "viva", "assessment"],
  deadline: ["deadline", "due tomorrow", "submission", "assignment due", "hand in"],
  // "interview" was listed here, so the extractor displayed "presentation"
  // while its own evidence span read "interview". A category that contradicts
  // its evidence is worse than no category.
  presentation: ["presentation", "present to", "defence", "defense"],
  interview: ["interview", "technical round", "hiring", "recruiter"],
  work: ["work", "shift", "office", "meeting", "project"],
  study: ["study", "studying", "revision", "revising", "coursework", "lecture", "class"],
  travel: ["flight", "train", "travel", "journey", "commute", "trip"],
  family: ["family", "mum", "mom", "dad", "parents", "sister", "brother", "partner"],
  "social event": ["party", "wedding", "dinner with", "night out", "birthday"],
  conflict: ["argument", "argued", "fight", "fell out", "disagreement"],
  bereavement: ["funeral", "passed away", "died"],
  finances: ["rent", "bills", "money", "debt", "loan"],
};

const CONTEXT_LEXICON = {
  sleep: {
    poor: ["couldn't sleep", "could not sleep", "can't sleep", "no sleep",
           "barely slept", "hardly slept", "didn't sleep", "insomnia",
           "awake all night", "restless night", "slept badly", "poor sleep",
           "up all night"],
    good: ["slept well", "good sleep", "well rested", "well-rested", "slept great"],
  },
  workload: {
    high: ["busy", "swamped", "overloaded", "so much work", "workload",
           "lots to do", "piled up", "back to back", "deadline", "too much on"],
    low: ["quiet day", "nothing much on", "light day", "free day", "not much to do"],
  },
  social: {
    isolated: ["alone", "lonely", "by myself", "on my own", "nobody to", "isolated"],
    "with others": ["with friends", "we went", "met up", "spent time with",
                    "together with", "dinner with", "out with", "saw my family",
                    "with a friend"],
  },
  activity: {
    low: ["didn't leave", "stayed in", "stayed home", "in bed all", "no exercise"],
    high: ["went for a run", "gym", "workout", "exercised", "walked a lot", "cycled"],
  },
  location: {
    home: ["at home", "stayed home", "from home", "in my room"],
    work: ["at work", "in the office", "at the office"],
    campus: ["on campus", "at uni", "at university", "in the library"],
    hospital: ["at the hospital", "in hospital", "at the clinic"],
  },
};

const TIME_PATTERNS = [
  [/\bday after tomorrow\b/, "day after tomorrow"], [/\btomorrow\b/, "tomorrow"],
  [/\byesterday\b/, "yesterday"], [/\blast night\b/, "last night"],
  [/\btonight\b/, "tonight"], [/\bthis morning\b/, "this morning"],
  [/\bthis evening\b/, "this evening"], [/\bnext week\b/, "next week"],
  [/\blast week\b/, "last week"], [/\btoday\b/, "today"],
];

const WORDNUM = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6,
                  seven: 7, eight: 8, nine: 9, ten: 10 };
const SLEEP_HOURS =
  /\b(?:slept|sleep|got)\s+(?:only\s+)?(?:about\s+)?(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:hours|hrs|h)\b/;
export const POOR_SLEEP_HOURS = 6;

/** Explicit first-person statements of feeling. See the note in detect.py. */
const SELF_REPORT = [
  [/\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?(stressed|stressed out|under pressure|overwhelmed|burnt out|burned out|exhausted|drained)\b/, "stress"],
  [/\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?(anxious|nervous|worried|scared|afraid|on edge)\b/, "anxiety"],
  [/\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?(sad|down|low|depressed|miserable|hopeless|lonely)\b/, "sadness"],
  [/\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?(angry|furious|annoyed|irritated|frustrated|fed up)\b/, "anger"],
  [/\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?(happy|great|good|excited|glad|delighted|cheerful)\b/, "joy"],
  [/\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?(grateful|thankful)\b/, "gratitude"],
  [/\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?(calm|relaxed|fine|ok|okay|settled|at ease)\b/, "calm"],
];

/**
 * Constructions that REVERSE or CANCEL a following self-statement.
 * Without these, "I am not sure I am good enough" matched "i am good" and the
 * interface reported JOY for a sentence expressing self-doubt.
 */
const NEGATORS = /\b(not|never|hardly|barely|no longer|don'?t|doesn'?t|didn'?t|can'?t|cannot|isn'?t|aren'?t|wasn'?t|won'?t)\b/;

/** Past or future framing: "yesterday I was anxious" is not a current state. */
const PAST = /\b(yesterday|last night|last week|earlier|used to|i was|had been|this morning)\b/;
const FUTURE = /\b(tomorrow|next week|will be|going to|expect to|i might|i may|probably)\b/;

/**
 * A feeling the person states OUTRIGHT, or null when the text does not.
 *
 * Three failures this now refuses, all of which the previous version got wrong
 * because it took the LONGEST match anywhere in the string:
 *
 *   "I am not sure I am good enough"          -> null      (was: joy)
 *   "I am grateful, but I am anxious"         -> anxiety    (was: gratitude)
 *   "Yesterday I was anxious, now relieved"   -> null       (was: anxiety)
 *
 * The last surviving statement wins, because a narrative ends on its current
 * state; an earlier clause is not the person's present feeling.
 */
export function selfReportedEmotion(text, { requirePresent = true } = {}) {
  const low = (text || "").toLowerCase();
  const hits = [];
  for (const [re, label] of SELF_REPORT) {
    const g = new RegExp(re.source, "g");
    let m;
    while ((m = g.exec(low)) !== null) {
      // look back over this clause only, not the whole sentence
      const before = low.slice(Math.max(0, m.index - 60), m.index);
      const clause = before.split(",").pop().split(".").pop();
      if (NEGATORS.test(clause)) continue;
      if (requirePresent && (PAST.test(clause) || FUTURE.test(clause))) continue;
      hits.push([m.index, label, m[0]]);
    }
  }
  if (!hits.length) return null;
  hits.sort((a, b) => a[0] - b[0]);
  const last = hits[hits.length - 1];
  return [last[1], last[2]];
}

const longestHit = (low, phrases) => {
  let best = null;
  for (const p of phrases) if (low.includes(p) && (!best || p.length > best.length)) best = p;
  return best;
};

export function extractContext(text, userFields = null) {
  const low = ` ${(text || "").toLowerCase().trim()} `;
  const out = {};

  const m = low.match(SLEEP_HOURS);
  if (m) {
    const h = WORDNUM[m[1]] ?? parseFloat(m[1]);
    if (isFinite(h)) {
      out.sleep = P(h < POOR_SLEEP_HOURS ? "poor" : "good", FieldSource.EXTRACTED,
                    0.9, `${m[0].trim()} (${h}h, threshold ${POOR_SLEEP_HOURS}h)`);
    }
  }

  for (const [field, values] of Object.entries(CONTEXT_LEXICON)) {
    if (field === "sleep" && known(out.sleep)) continue;
    let hv = null, hp = null;
    for (const [canon, phrases] of Object.entries(values)) {
      const hit = longestHit(low, phrases);
      if (hit && (!hp || hit.length > hp.length)) { hv = canon; hp = hit; }
    }
    out[field] = hv ? P(hv, FieldSource.EXTRACTED, 0.8, hp.trim()) : UNKNOWN();
  }

  let ev = null, ep = null;
  for (const [canon, phrases] of Object.entries(EVENT_LEXICON)) {
    const hit = longestHit(low, phrases);
    if (hit && (!ep || hit.length > ep.length)) { ev = canon; ep = hit; }
  }
  out.event = ev ? P(ev, FieldSource.EXTRACTED, 0.75, ep.trim()) : UNKNOWN();

  out.time_context = UNKNOWN();
  for (const [re, canon] of TIME_PATTERNS) {
    const t = low.match(re);
    if (t) { out.time_context = P(canon, FieldSource.EXTRACTED, 0.95, t[0]); break; }
  }

  for (const [k, v] of Object.entries(userFields || {})) {
    if (v == null || v === "" || v === "unknown") continue;
    out[k] = P(v, FieldSource.USER_REPORTED, 1.0, "check-in field");
  }
  return out;
}

/* -------------------------------------------------------- emotion (local) */
const LEXICON = {
  anxiety: ["anxious", "anxiety", "nervous", "worried", "worry", "scared", "afraid",
            "panic", "dread", "uneasy", "on edge"],
  stress: ["stressed", "stress", "pressure", "overwhelmed", "overloaded",
           "burnt out", "burned out", "exhausted", "drained", "tense"],
  sadness: ["sad", "down", "depressed", "unhappy", "miserable", "crying", "cried",
            "hopeless", "lonely", "empty"],
  anger: ["angry", "furious", "annoyed", "irritated", "frustrated", "mad", "fed up"],
  joy: ["happy", "great", "excited", "delighted", "glad", "wonderful", "pleased",
        "lovely", "enjoyed"],
  gratitude: ["grateful", "thankful", "appreciate", "thanks"],
  calm: ["calm", "relaxed", "peaceful", "settled", "at ease", "fine"],
  confusion: ["confused", "unsure", "uncertain", "puzzled", "lost"],
};

/** The BASELINE. Named so at every call site; never presented as the model. */
export function lexiconEmotion(text) {
  const low = ` ${(text || "").toLowerCase()} `;
  const scores = {};
  for (const [label, words] of Object.entries(LEXICON)) {
    const hits = words.filter((w) => low.includes(w)).length;
    if (hits) scores[label] = Math.min(0.35 + 0.15 * hits, 0.9);
  }
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  if (!ranked.length) return { label: "neutral", score: 0.3, backend: "lexicon" };
  return { label: ranked[0][0], score: ranked[0][1], backend: "lexicon",
           runnerUp: ranked[1] || null };
}

/* -------------------------------------------------------------- events */
export function newEventId(pid, ts, text) {
  let h = 2166136261;
  for (const ch of `${pid}|${ts}|${text}`) {
    h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0;
  }
  return `ev_${h.toString(16).padStart(8, "0")}${(text.length % 997).toString(16)}`;
}

/**
 * Build one event.
 *
 * The MODEL is the feeling. An explicit statement is recorded BESIDE it, in
 * `statedEmotion`, and never instead of it. The old precedence let a regex
 * silently overwrite the Transformer, which is how "I am not sure I am good
 * enough" was reported as joy. A field the person filled in explicitly still
 * wins: that is a real self-report through a control, not a substring match.
 */
export function buildEvent(text, { personId, timestamp, userFields, prediction,
                                   dataStatus = "USER" } = {}) {
  const ts = timestamp || new Date().toISOString();
  const ctx = extractContext(text, userFields);
  const stated = selfReportedEmotion(text);
  const pred = prediction || lexiconEmotion(text);

  let emotion;
  if (userFields && userFields.emotion) {
    emotion = P(userFields.emotion, FieldSource.USER_REPORTED, 1.0, "check-in field");
  } else {
    emotion = P(pred.label, FieldSource.MODEL, pred.score,
                `${pred.backend}${pred.rawLabel ? ":" + pred.rawLabel : ""}`);
  }
  const statedEmotion = stated
    ? P(stated[0], FieldSource.EXTRACTED, 0.9, `you wrote: "${stated[1]}"`)
    : P(null, FieldSource.UNKNOWN);

  const ev = { eventId: newEventId(personId, ts, text), personId, timestamp: ts,
               rawText: text, dataStatus, backend: pred.backend,
               modelVersion: pred.model || pred.backend, corrections: [] };
  for (const f of CONTEXT_FIELDS) {
    ev[f] = f === "emotion" ? emotion
          : f === "statedEmotion" ? statedEmotion
          : (ctx[f] || UNKNOWN());
  }
  return ev;
}

export const isKnown = known;
export const knownFields = (ev) =>
  Object.fromEntries(CONTEXT_FIELDS.filter((f) => known(ev[f])).map((f) => [f, ev[f]]));
export const unknownFields = (ev) => CONTEXT_FIELDS.filter((f) => !known(ev[f]));

export function correctField(ev, field, value) {
  return { ...ev, [field]: P(value, FieldSource.CORRECTED, 1.0, "user correction"),
           corrections: [...new Set([...ev.corrections, field])] };
}

/* -------------------------------------------------------- hypergraph */
const FIELD_TYPES = { emotion: "Emotion", event: "Event", time_context: "Time",
                      sleep: "Sleep", activity: "Activity", social: "Social",
                      workload: "Workload", location: "Location" };

export function buildEventHypergraph(personId, events) {
  const vertices = new Map();
  const edges = [];
  const add = (key, type, value) => {
    if (!vertices.has(key)) vertices.set(key, { key, type, value });
    return key;
  };
  for (const ev of [...events].sort((a, b) => a.timestamp.localeCompare(b.timestamp))) {
    const keys = [add(`Person=${ev.personId}`, "Person", ev.personId)];
    const sources = {};
    for (const [f, type] of Object.entries(FIELD_TYPES)) {
      if (!known(ev[f])) continue;                 // absence is not a vertex
      keys.push(add(`${type}=${ev[f].value}`, type, String(ev[f].value)));
      sources[f] = ev[f].source;
    }
    edges.push({ edgeId: ev.eventId, vertices: keys, timestamp: ev.timestamp,
                 personId: ev.personId, rawText: ev.rawText,
                 dataStatus: ev.dataStatus, sources, arity: keys.length });
  }
  const pairs = new Map();
  for (const e of edges) {
    const vs = e.vertices.filter((v) => !v.startsWith("Person=")).sort();
    for (let i = 0; i < vs.length; i++)
      for (let j = i + 1; j < vs.length; j++) {
        const k = `${vs[i]}||${vs[j]}`;
        pairs.set(k, (pairs.get(k) || 0) + 1);
      }
  }
  const recurring = [...pairs.entries()].filter(([, n]) => n >= 2)
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => ({ a: k.split("||")[0], b: k.split("||")[1], n }));

  const byType = {};
  for (const v of vertices.values()) byType[v.type] = (byType[v.type] || 0) + 1;
  const arities = edges.map((e) => e.arity);

  return {
    personId, vertices: [...vertices.values()], edges, recurring,
    summary: {
      nVertices: vertices.size, nEdges: edges.length,
      meanArity: arities.length ? arities.reduce((a, b) => a + b, 0) / arities.length : 0,
      maxArity: arities.length ? Math.max(...arities) : 0,
      verticesByType: byType, recurringPairs: recurring.length,
      syntheticEdges: edges.filter((e) => e.dataStatus !== "USER").length,
    },
  };
}

/* ------------------------------------------------------------- the twin */
export const MIN_EPISODES_FOR_PATTERN = 3;
export const FIELD_WEIGHTS = { event: 3.0, sleep: 2.5, workload: 2.0, social: 1.5,
                               activity: 1.0, location: 1.0, time_context: 0.5 };

export class PersonalTwin {
  constructor(personId = "You", dataStatus = "USER") {
    this.personId = personId;
    this.dataStatus = dataStatus;
    this.events = [];
  }

  addEvent(ev) {
    if (this.events.some((e) => e.eventId === ev.eventId)) return;
    this.events.push(ev);
    this.events.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  clear() { this.events = []; }

  get isSynthetic() { return this.events.some((e) => e.dataStatus !== "USER"); }

  /** Past episodes sharing context, best first, each with its reason. */
  similarEpisodes(ev, { topK = 5, minScore = 1.5 } = {}) {
    const out = [];
    for (const past of this.events) {
      if (past.eventId === ev.eventId || past.timestamp > ev.timestamp) continue;
      const matched = [];
      let score = 0;
      for (const [f, w] of Object.entries(FIELD_WEIGHTS)) {
        if (known(ev[f]) && known(past[f]) && ev[f].value === past[f].value) {
          matched.push([f, String(ev[f].value)]);
          score += w;
        }
      }
      // the conjunction is worth more than the sum of its parts
      if (matched.length >= 2) score *= 1 + 0.15 * (matched.length - 1);
      if (score >= minScore) {
        out.push({ eventId: past.eventId, timestamp: past.timestamp,
                   emotion: known(past.emotion) ? past.emotion.value : null,
                   score, matched, rawText: past.rawText,
                   explanation: matched.length
                     ? "same " + matched.map(([f, v]) => `${FIELD_LABELS[f] || f} (${v})`).join(", ")
                     : "no fields in common" });
      }
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, topK);
  }

  /** A tendency, or an explicit refusal. The floor is not negotiable. */
  patternInsight(ev, minEpisodes = MIN_EPISODES_FOR_PATTERN) {
    const similar = this.similarEpisodes(ev, { topK: 25, minScore: 1.5 });
    if (similar.length < minEpisodes) {
      return {
        sufficient: false, nSimilar: similar.length,
        statement: `Your Digital Twin is still learning. It found ${similar.length} `
          + `comparable past ${similar.length === 1 ? "episode" : "episodes"}, and needs `
          + `at least ${minEpisodes} before describing a pattern.`,
        caveats: ["A pattern stated from one or two episodes would be noise, not a pattern."],
      };
    }
    const emotions = similar.map((s) => s.emotion).filter(Boolean);
    if (!emotions.length) {
      return { sufficient: false, nSimilar: similar.length,
               statement: "Comparable episodes were found, but none carries a recorded "
                 + "feeling, so no tendency can be described.", caveats: [] };
    }
    const counts = {};
    for (const e of emotions) counts[e] = (counts[e] || 0) + 1;
    const [dominant, nDom] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    const shared = {};
    for (const s of similar) for (const [f, v] of s.matched) {
      const k = `${FIELD_LABELS[f] || f}: ${v}`;
      shared[k] = (shared[k] || 0) + 1;
    }
    const context = Object.entries(shared).filter(([, n]) => n >= minEpisodes)
      .sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k]) => k);

    return {
      sufficient: true, nSimilar: similar.length, nSupporting: nDom,
      dominantEmotion: dominant, matchedContext: context,
      statement: `Across ${similar.length} comparable past episodes`
        + (context.length ? ` (${context.join(", ")})` : "")
        + `, the most frequently recorded feeling was ${dominant}, in ${nDom} of them.`,
      caveats: [
        "This is an association in your own history, not a cause.",
        "It describes what was recorded before, not what will happen.",
        `Based on ${similar.length} episodes, which is a small sample.`,
      ],
    };
  }

  profile() {
    if (!this.events.length) return { personId: this.personId, nEvents: 0 };
    const recent = this.events[this.events.length - 1];
    const emo = {}, fields = {};
    for (const e of this.events) {
      if (known(e.emotion)) emo[e.emotion.value] = (emo[e.emotion.value] || 0) + 1;
      for (const [f, p] of Object.entries(knownFields(e))) {
        if (f === "emotion") continue;
        fields[f] = fields[f] || {};
        fields[f][p.value] = (fields[f][p.value] || 0) + 1;
      }
    }
    const recurring = {};
    for (const [f, counts] of Object.entries(fields)) {
      const top = Object.entries(counts).filter(([, n]) => n >= 2)
        .sort((a, b) => b[1] - a[1]).slice(0, 3);
      if (top.length) recurring[f] = top.map(([value, n]) => ({ value, n }));
    }
    return {
      personId: this.personId, nEvents: this.events.length,
      isSynthetic: this.isSynthetic,
      firstEvent: this.events[0].timestamp, lastEvent: recent.timestamp,
      recentEmotion: known(recent.emotion) ? recent.emotion.value : null,
      recentText: recent.rawText,
      emotionCounts: Object.fromEntries(Object.entries(emo).sort((a, b) => b[1] - a[1])),
      recurringContext: recurring,
      readyForPatterns: this.events.length >= MIN_EPISODES_FOR_PATTERN,
    };
  }

  /**
   * How far personalisation is actually supported for this person, right now.
   *
   * THIS IS AN EXPERIMENTAL MECHANISM, NOT A VALIDATED RESULT. The thresholds
   * below are engineering defaults chosen so the interface refuses to imply
   * personal knowledge it does not have. The scientific question -- whether an
   * evidence gate beats simply counting observations -- has NOT been tested,
   * and the interface must say so wherever this is displayed.
   *
   * What IS measured, on 218 participants over 4.8 years, is the motivation:
   * per-person predictability ranges from r = -0.24 to 0.69, so treating every
   * user as equally personalisable is demonstrably wrong.
   *
   * Every number returned here comes from the stored history. Nothing is
   * assumed and nothing is hard-coded per user.
   */
  personalisationStatus(query = null) {
    const n = this.events.length;
    const relevant = query ? this.similarEpisodes(query, { topK: 50, minScore: 1.5 }).length : 0;

    let level, headline, action;
    if (n < MIN_EPISODES_FOR_PATTERN) {
      level = "insufficient";
      headline = "Not enough history to personalise";
      action = "Using a population-informed estimate";
    } else if (query && relevant < 2) {
      level = "emerging";
      headline = "Some history, but little of it comparable to this situation";
      action = "Mostly population-informed";
    } else if (n < 10) {
      level = "emerging";
      headline = "Personal pattern is starting to form";
      action = "Blending personal history with population patterns";
    } else {
      level = "supported";
      headline = "Personal history is informing this estimate";
      action = "Using your own history";
    }

    // Reasons are statements about the stored data, never invented text.
    const reasons = [
      `${n} observation${n === 1 ? "" : "s"} recorded`,
      ...(query ? [`${relevant} comparable past episode${relevant === 1 ? "" : "s"}`] : []),
      ...(n < MIN_EPISODES_FOR_PATTERN
          ? [`at least ${MIN_EPISODES_FOR_PATTERN} are needed before any pattern is reported`]
          : []),
    ];
    return { level, headline, action, reasons, nEvents: n, nRelevant: relevant,
             experimental: true };
  }

  toJSON() {
    return { personId: this.personId, dataStatus: this.dataStatus, events: this.events };
  }

  static fromJSON(d) {
    const t = new PersonalTwin(d.personId, d.dataStatus);
    t.events = d.events || [];
    return t;
  }
}
