import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AuthProvider, useAuth } from "@/components/auth-provider";
import { notifyUnauthorized, setStoredToken } from "@/lib/auth-storage";
import type { UserOut } from "@/lib/types";

const mockUser: UserOut = { id: 1, email: "alice@example.com", is_active: true, created_at: null };

// `AuthProvider` imports these three directly from lib/api.ts — mocked
// here so this test suite exercises the PROVIDER's own state-machine
// logic (mount-time restore / login / register / logout / 401 reset),
// independent of lib/api.ts's own request() behavior (already covered
// by tests/lib-api.test.ts).
vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getMe: vi.fn(),
    login: vi.fn(),
    registerAccount: vi.fn(),
  };
});

import { getMe, login as apiLogin, registerAccount } from "@/lib/api";

function Probe() {
  const { user, login, register, logout } = useAuth();
  return (
    <div>
      <span data-testid="user-state">
        {user === undefined ? "loading" : user === null ? "signed-out" : user.email}
      </span>
      <button onClick={() => login("alice@example.com", "hunter2pass")}>login</button>
      <button onClick={() => register("alice@example.com", "hunter2pass")}>register</button>
      <button onClick={() => logout()}>logout</button>
    </div>
  );
}

describe("AuthProvider", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.mocked(getMe).mockReset();
    vi.mocked(apiLogin).mockReset();
    vi.mocked(registerAccount).mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("starts in the loading state, then resolves to signed-out with no stored token", async () => {
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("signed-out"));
    expect(getMe).not.toHaveBeenCalled();
  });

  it("restores the session from a stored token on mount", async () => {
    setStoredToken("existing-token");
    vi.mocked(getMe).mockResolvedValueOnce(mockUser);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("alice@example.com"));
  });

  it("falls back to signed-out if restoring a stored token fails", async () => {
    setStoredToken("stale-token");
    vi.mocked(getMe).mockRejectedValueOnce(new Error("network error"));

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("signed-out"));
  });

  it("login() stores the token and sets the user", async () => {
    const userEventInstance = userEvent.setup();
    vi.mocked(apiLogin).mockResolvedValueOnce({
      access_token: "new-token",
      token_type: "bearer",
      expires_in_minutes: 60,
    });
    vi.mocked(getMe).mockResolvedValueOnce(mockUser);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("signed-out"));

    await userEventInstance.click(screen.getByText("login"));

    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("alice@example.com"));
    expect(window.localStorage.getItem("veridoc.auth.token")).toBe("new-token");
  });

  it("register() stores the token and sets the user", async () => {
    const userEventInstance = userEvent.setup();
    vi.mocked(registerAccount).mockResolvedValueOnce({
      access_token: "fresh-token",
      token_type: "bearer",
      expires_in_minutes: 60,
    });
    vi.mocked(getMe).mockResolvedValueOnce(mockUser);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("signed-out"));

    await userEventInstance.click(screen.getByText("register"));

    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("alice@example.com"));
  });

  it("logout() clears the token and resets user to signed-out", async () => {
    const userEventInstance = userEvent.setup();
    setStoredToken("existing-token");
    vi.mocked(getMe).mockResolvedValueOnce(mockUser);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("alice@example.com"));

    await userEventInstance.click(screen.getByText("logout"));

    expect(screen.getByTestId("user-state")).toHaveTextContent("signed-out");
    expect(window.localStorage.getItem("veridoc.auth.token")).toBeNull();
  });

  it("resets to signed-out when notifyUnauthorized fires mid-session (e.g. a 401 elsewhere in the app)", async () => {
    setStoredToken("existing-token");
    vi.mocked(getMe).mockResolvedValueOnce(mockUser);

    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("user-state")).toHaveTextContent("alice@example.com"));

    act(() => {
      notifyUnauthorized();
    });

    expect(screen.getByTestId("user-state")).toHaveTextContent("signed-out");
  });

  it("useAuth() throws when used outside AuthProvider", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<Probe />)).toThrow("useAuth() must be used within <AuthProvider>.");
    consoleError.mockRestore();
  });
});