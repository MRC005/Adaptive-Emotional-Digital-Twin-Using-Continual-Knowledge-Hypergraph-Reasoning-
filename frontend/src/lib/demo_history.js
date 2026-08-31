/**
 * The synthetic demonstration history, mirrored from
 * aedt/twin/demo_history.py so the Python and browser demos tell the same story.
 *
 * SYNTHETIC. A fictional person. Every event built from this is stamped
 * SYNTHETIC_DEMO, the twin is marked synthetic, and the interface labels it.
 *
 * Three recurring situations are built in, so retrieval has something true to
 * find: deadline/exam + poor sleep -> stress; appointment + poor sleep ->
 * anxiety; social evening + good sleep -> joy. Filler episodes are included so
 * retrieval has to discriminate rather than return everything.
 */
export const DEMO_PERSON_ID = "Demo_User";

export const DEMO_SCRIPT = [
  {
    "daysAgo": 86,
    "text": "First week back. Quiet so far, nothing much on.",
    "fields": {}
  },
  {
    "daysAgo": 82,
    "text": "Went for a long run this morning and slept well last night. Feeling good.",
    "fields": {}
  },
  {
    "daysAgo": 78,
    "text": "I am stressed. Deadline for the group assignment tomorrow and I barely slept.",
    "fields": {
      "workload": "high"
    }
  },
  {
    "daysAgo": 74,
    "text": "Had dinner with friends, really enjoyed it. Slept well afterwards.",
    "fields": {}
  },
  {
    "daysAgo": 71,
    "text": "I couldn't sleep again. Hospital appointment tomorrow for my scan.",
    "fields": {}
  },
  {
    "daysAgo": 70,
    "text": "The appointment went fine in the end. Relieved.",
    "fields": {}
  },
  {
    "daysAgo": 65,
    "text": "I am exhausted. Two deadlines this week and I slept about four hours.",
    "fields": {
      "workload": "high"
    }
  },
  {
    "daysAgo": 61,
    "text": "Quiet day, stayed home and read.",
    "fields": {}
  },
  {
    "daysAgo": 58,
    "text": "I am anxious about the exam tomorrow. Couldn't sleep at all.",
    "fields": {
      "workload": "high"
    }
  },
  {
    "daysAgo": 54,
    "text": "Went to the gym, slept well, feeling much better today.",
    "fields": {}
  },
  {
    "daysAgo": 49,
    "text": "Another hospital appointment tomorrow. Didn't sleep well thinking about it.",
    "fields": {}
  },
  {
    "daysAgo": 45,
    "text": "Saw my family at the weekend, it was lovely.",
    "fields": {}
  },
  {
    "daysAgo": 40,
    "text": "I am stressed and overwhelmed. Presentation tomorrow and I hardly slept.",
    "fields": {
      "workload": "high"
    }
  },
  {
    "daysAgo": 36,
    "text": "Nothing much on today. Watched a film.",
    "fields": {}
  },
  {
    "daysAgo": 31,
    "text": "I feel low. Argued with my flatmate and slept badly.",
    "fields": {}
  },
  {
    "daysAgo": 27,
    "text": "Long walk with a friend, slept well. Feeling calm.",
    "fields": {}
  },
  {
    "daysAgo": 22,
    "text": "I am stressed. Coursework due tomorrow, worked until 2am, four hours sleep.",
    "fields": {
      "workload": "high"
    }
  },
  {
    "daysAgo": 18,
    "text": "Hospital appointment tomorrow again. I couldn't sleep.",
    "fields": {}
  },
  {
    "daysAgo": 14,
    "text": "Results came back clear. I feel grateful.",
    "fields": {}
  },
  {
    "daysAgo": 9,
    "text": "Busy week at work but I slept well and it was manageable.",
    "fields": {}
  },
  {
    "daysAgo": 5,
    "text": "I am tired but okay. Quiet weekend, saw nobody.",
    "fields": {}
  },
  {
    "daysAgo": 2,
    "text": "Went out for a birthday dinner with friends. Slept well. Really enjoyed it.",
    "fields": {}
  }
];

/** Build the history by running the REAL perception path over each line. */
export async function buildDemoHistory(buildOne, now = new Date()) {
  const out = [];
  for (const { daysAgo, text, fields } of DEMO_SCRIPT) {
    const d = new Date(now.getTime() - daysAgo * 86400000);
    d.setHours(20, 0, 0, 0);
    out.push(await buildOne(text, d.toISOString(), fields));
  }
  return out;
}
