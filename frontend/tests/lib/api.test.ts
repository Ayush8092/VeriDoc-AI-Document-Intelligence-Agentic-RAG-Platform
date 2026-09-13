import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// `lib/api.ts` reads the stored token via `lib/auth-storage.ts` on every
// request and notifies subscribers on a 401 — tested here against the
// REAL auth-storage module (not mocked), so this test suite is actually
// exercising the real integration between the two, not just api.ts's
// code in isolation.
import { getHealth, ask, ApiError } from "@/lib/api";
import { clearStoredToken, getStoredToken, onUnauthorized, setStoredToken } from "@/lib/auth-storage";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("lib/api.ts request()", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not attach an Authorization header when no token is stored", async () => {
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(jsonResponse({ status: "ok" }));

    await getHealth();

    const [, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const headers = new Headers(init?.headers);
    expect(headers.has("authorization")).toBe(false);
  });

  it("attaches Authorization: Bearer <token> on every request when a session exists", async () => {
    setStoredToken("my-real-token");
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(jsonResponse({ status: "ok" }));

    await getHealth();

    const [, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe("Bearer my-real-token");
  });

  it("resolves with the parsed JSON body on a 2xx response", async () => {
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(jsonResponse({ status: "ok" }));

    const result = await getHealth();

    expect(result).toEqual({ status: "ok" });
  });

  it("sends the question in the POST body for ask()", async () => {
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      jsonResponse({ answer: "42", found: true, citations: [], query_type: null, claims: [] }),
    );

    await ask("What is the answer?");

    const [url, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(String(url)).toContain("/ask");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init!.body as string)).toEqual({ question: "What is the answer?" });
  });

  it("throws ApiError with the string `detail` field from a JSON error body", async () => {
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      jsonResponse({ detail: "Invalid credentials." }, 401),
    );

    await expect(getHealth()).rejects.toMatchObject({
      message: "Invalid credentials.",
      status: 401,
    });
  });

  it("joins a FastAPI/pydantic validation-error array `detail` into one message", async () => {
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      jsonResponse(
        { detail: [{ msg: "field required" }, { msg: "value is not a valid email" }] },
        422,
      ),
    );

    await expect(getHealth()).rejects.toMatchObject({
      message: "field required; value is not a valid email",
      status: 422,
    });
  });

  it("falls back to statusText when the error body is not JSON", async () => {
    const res = new Response("<html>Internal Server Error</html>", {
      status: 500,
      statusText: "Internal Server Error",
      headers: { "content-type": "text/html" },
    });
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(res);

    await expect(getHealth()).rejects.toMatchObject({
      message: "Internal Server Error",
      status: 500,
    });
  });

  it("throws a network-style ApiError (status 0) when fetch itself rejects", async () => {
    (fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(getHealth()).rejects.toMatchObject({ status: 0 });
    await expect(getHealth()).rejects.toBeInstanceOf(ApiError);
  });

  it("clears the stored token and notifies subscribers on a 401 for an authenticated request", async () => {
    setStoredToken("stale-token");
    const listener = vi.fn();
    const unsubscribe = onUnauthorized(listener);
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(jsonResponse({ detail: "Not authenticated" }, 401));

    await expect(getHealth()).rejects.toMatchObject({ status: 401 });

    expect(getStoredToken()).toBeNull();
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("does NOT fire the unauthorized notification for a 401 when no token was attached", async () => {
    // No token stored — this project's tenant model treats anonymous
    // requests as valid (shared public corpus), so a 401 with no token
    // present is not "your session expired", it's just an ordinary
    // auth-required response; there is no session to invalidate.
    const listener = vi.fn();
    const unsubscribe = onUnauthorized(listener);
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(jsonResponse({ detail: "Not authenticated" }, 401));

    await expect(getHealth()).rejects.toMatchObject({ status: 401 });

    expect(listener).not.toHaveBeenCalled();
    unsubscribe();
  });

  it("does not clear the token or notify subscribers on a non-401 error", async () => {
    setStoredToken("still-valid-token");
    const listener = vi.fn();
    const unsubscribe = onUnauthorized(listener);
    (fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(jsonResponse({ detail: "Not found" }, 404));

    await expect(getHealth()).rejects.toMatchObject({ status: 404 });

    expect(getStoredToken()).toBe("still-valid-token");
    expect(listener).not.toHaveBeenCalled();
    unsubscribe();
  });
});