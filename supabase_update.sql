alter table public.readings add column if not exists detected_device text;
alter table public.readings add column if not exists detection_confidence real;
alter table public.readings add column if not exists raw_device text;
alter table public.readings add column if not exists raw_confidence real;
alter table public.readings add column if not exists detection_age_s real;
alter table public.readings add column if not exists inference_ms integer;

notify pgrst, 'reload schema';
