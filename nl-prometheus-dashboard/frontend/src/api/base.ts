const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers ?? {})
      }
    });
  } catch (error) {
    const target = API_BASE_URL || "the frontend dev proxy at /api";
    throw new Error(`Failed to reach backend through ${target}. ${error instanceof Error ? error.message : ""}`.trim());
  }

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}
