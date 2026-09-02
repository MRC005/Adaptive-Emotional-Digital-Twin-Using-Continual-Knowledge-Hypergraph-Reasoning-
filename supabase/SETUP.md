# Enabling accounts

The application works fully without this. Unconfigured, it runs local-only and
anonymous, which is how the panel demonstration is designed to run. Follow this
only if you want histories to persist across devices and sessions.

## 1. Create the project

1. <https://supabase.com> → **New project**. Any region; the free tier is enough.
2. Wait for provisioning (~2 minutes).

## 2. Create the table

**SQL Editor → New query** → paste all of `supabase/schema.sql` → **Run**.
It should report success with no rows returned.

## 3. Turn off email confirmation (for a demonstration)

**Authentication → Providers → Email** → turn **Confirm email** OFF.

With it on, a new account cannot save anything until the person clicks a link in
their inbox — which is not something to discover in front of a panel. The app
handles that case and says so, but off is the right setting for a demo.

## 4. Copy the two values

**Project Settings → API**:

- **Project URL** → `VITE_SUPABASE_URL`
- **anon / public** key → `VITE_SUPABASE_ANON_KEY`

The anon key is *designed* to be public and is safe in a browser bundle; it
grants nothing on its own, because row-level security is what decides access.
**Never put the `service_role` key here** — it bypasses RLS entirely.

## 5. Configure

Local: create `frontend/.env.local`

```
VITE_SUPABASE_URL=https://YOUR-PROJECT.supabase.co
VITE_SUPABASE_ANON_KEY=YOUR-ANON-KEY
```

Vercel: **Project Settings → Environment Variables**, add both, redeploy.
`.env.local` is gitignored and must stay that way.

## 6. Verify

Sign up, add a check-in, sign out, sign back in — the history should return.
Then open a private window, create a second account, and confirm it sees an
empty history. That second check is the one that matters.
