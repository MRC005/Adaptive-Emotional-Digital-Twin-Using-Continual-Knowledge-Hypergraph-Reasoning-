-- Persistent personal twins: schema and row-level security.
--
-- Run once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
--
-- THE SECURITY MODEL IS HERE, NOT IN THE BROWSER. The client never sends a user
-- id. `user_id` defaults to auth.uid() -- the id inside the verified JWT -- and
-- the policies below restrict every operation to rows where user_id = auth.uid().
-- A client that forged a different id in a request body would still only be able
-- to write a row owned by itself, and could not read anyone else's at all.
--
-- These check-ins are a person's own words about how they feel. That is
-- sensitive, so the default is deny: no anonymous access, no cross-user access,
-- and no analytics reads. The research pipeline never touches this table.

create table if not exists public.twin_events (
  id          bigserial primary key,
  user_id     uuid not null references auth.users (id) on delete cascade
                       default auth.uid(),
  event_id    text not null,
  occurred_at timestamptz not null,
  payload     jsonb not null,
  created_at  timestamptz not null default now(),

  -- re-submitting the same check-in must not duplicate it
  unique (user_id, event_id)
);

create index if not exists twin_events_user_time_idx
  on public.twin_events (user_id, occurred_at);

alter table public.twin_events enable row level security;

-- Four explicit policies. `with check` governs what may be written; `using`
-- governs what may be seen. Both are pinned to auth.uid().
drop policy if exists twin_events_select_own on public.twin_events;
create policy twin_events_select_own on public.twin_events
  for select using (auth.uid() = user_id);

drop policy if exists twin_events_insert_own on public.twin_events;
create policy twin_events_insert_own on public.twin_events
  for insert with check (auth.uid() = user_id);

drop policy if exists twin_events_update_own on public.twin_events;
create policy twin_events_update_own on public.twin_events
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists twin_events_delete_own on public.twin_events;
create policy twin_events_delete_own on public.twin_events
  for delete using (auth.uid() = user_id);

-- Account deletion. GoTrue will not let an anon-key client delete a user, so
-- this runs as the definer -- but it can only ever delete auth.uid(), which is
-- taken from the verified token and cannot be supplied by the caller.
create or replace function public.delete_own_account()
returns void
language plpgsql
security definer
set search_path = public, auth
as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then
    raise exception 'not authenticated';
  end if;
  delete from public.twin_events where user_id = uid;
  delete from auth.users where id = uid;   -- cascades to any remaining rows
end;
$$;

revoke all on function public.delete_own_account() from public, anon;
grant execute on function public.delete_own_account() to authenticated;
