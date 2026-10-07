create table if not exists public.sessions (
  id text primary key,
  participant text not null,
  device text not null,
  distance_cm real not null,
  planned_minutes real,
  started_at timestamptz not null,
  ended_at timestamptz,
  samples integer not null default 0
);

create table if not exists public.readings (
  id bigint generated always as identity primary key,
  session_id text not null references public.sessions (id) on delete cascade,
  ts timestamptz not null,
  elapsed_s double precision not null,
  r integer not null,
  g integer not null,
  b integer not null,
  w integer not null,
  lux double precision not null,
  unique (session_id, elapsed_s)
);

create table if not exists public.detections (
  id bigint generated always as identity primary key,
  ts timestamptz not null unique,
  device text not null,
  confidence real not null,
  inference_ms integer
);

create index if not exists sessions_started_at_idx on public.sessions (started_at desc);

alter table public.sessions enable row level security;
alter table public.readings enable row level security;
alter table public.detections enable row level security;

grant select, insert, update, delete on public.sessions, public.readings, public.detections to service_role;
grant usage, select on all sequences in schema public to service_role;
