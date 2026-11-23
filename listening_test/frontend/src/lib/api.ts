export type NextResponse =
  | { done: true }
  | {
      done: false;
      sample_id: string;
      audio_url: string;
      total: number;
      current_round: number;
      completed_this_round: number;
    };

export type Profile = {
  expertise: 1 | 2 | 3;
  note: string;
};

const SESSION_KEY = "listening_test_session_id";
const PROFILE_KEY = "listening_test_profile";

function generateSessionId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = generateSessionId();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function loadProfile(): Profile | null {
  const raw = localStorage.getItem(PROFILE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if ([1, 2, 3].includes(parsed?.expertise) && typeof parsed?.note === "string") {
      return parsed as Profile;
    }
  } catch {

  }
  return null;
}

export function saveProfile(profile: Profile): void {
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function fetchNext(sessionId: string): Promise<NextResponse> {
  return fetch(`/api/next?session_id=${encodeURIComponent(sessionId)}`).then((res) =>
    asJson<NextResponse>(res)
  );
}

export function fetchStats(): Promise<{
  total: number;
  current_round: number;
  completed_this_round: number;
  pending_this_round: number;
  in_progress: number;
  descriptions_collected: number;
}> {
  return fetch("/api/stats").then((res) =>
    asJson<{
      total: number;
      current_round: number;
      completed_this_round: number;
      pending_this_round: number;
      in_progress: number;
      descriptions_collected: number;
    }>(res)
  );
}

export function skipSample(body: {
  session_id: string;
  sample_id: string;
}): Promise<NextResponse> {
  return fetch("/api/skip", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((res) => asJson<NextResponse>(res));
}

export function submitDescription(body: {
  session_id: string;
  sample_id: string;
  free_description: string;
  guided_description: string;
  guided_choices: Record<string, string> | null;
  describer_expertise: 1 | 2 | 3;
  describer_note: string | null;
}): Promise<{ ok: true }> {
  return fetch("/api/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((res) => asJson<{ ok: true }>(res));
}
