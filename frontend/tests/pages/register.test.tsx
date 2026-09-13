import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import RegisterPage from "@/app/register/page";
import { AuthProvider } from "@/components/auth-provider";

// Same integration shape as tests/pages/login.test.tsx and
// tests/components/auth-provider.test.tsx: mock the NETWORK boundary
// (lib/api.ts) and let the real AuthProvider run, so this suite
// exercises the actual page + provider wiring a user experiences —
// app/register/page.tsx calls useAuth().register(), which itself calls
// registerAccount() then getMe() (see components/auth-provider.tsx) —
// not a re-implementation of AuthProvider's own logic.
vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getMe: vi.fn(),
    login: vi.fn(),
    registerAccount: vi.fn(),
  };
});

import { getMe, registerAccount, ApiError } from "@/lib/api";

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

function renderRegisterPage() {
  return render(
    <AuthProvider>
      <RegisterPage />
    </AuthProvider>,
  );
}

describe("RegisterPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    pushMock.mockReset();
    vi.mocked(getMe).mockReset();
    vi.mocked(registerAccount).mockReset();
  });

  it("renders the create-account form with email and password fields", () => {
    renderRegisterPage();
    expect(screen.getByRole("heading", { name: "Create an account" })).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create account/i })).toBeInTheDocument();
  });

  it("marks email and password as required fields", () => {
    renderRegisterPage();
    expect(screen.getByLabelText("Email")).toBeRequired();
    expect(screen.getByLabelText("Password")).toBeRequired();
  });

  it("uses the correct input types and autocomplete hints", () => {
    renderRegisterPage();
    expect(screen.getByLabelText("Email")).toHaveAttribute("type", "email");
    expect(screen.getByLabelText("Email")).toHaveAttribute("autoComplete", "email");
    expect(screen.getByLabelText("Password")).toHaveAttribute("type", "password");
    // "new-password" (not "current-password", which login.test.tsx's
    // page uses) — a real, meaningful difference: it tells a password
    // manager to OFFER to generate/save a new credential here, not
    // autofill an existing one.
    expect(screen.getByLabelText("Password")).toHaveAttribute("autoComplete", "new-password");
  });

  it("sets minLength=8 on the password field, mirroring the backend's minimum", () => {
    renderRegisterPage();
    expect(screen.getByLabelText("Password")).toHaveAttribute("minLength", "8");
  });

  it("lets the user type into the email and password fields", async () => {
    const user = userEvent.setup();
    renderRegisterPage();

    const emailInput = screen.getByLabelText("Email") as HTMLInputElement;
    const passwordInput = screen.getByLabelText("Password") as HTMLInputElement;

    await user.type(emailInput, "alice@example.com");
    await user.type(passwordInput, "hunter2pass");

    expect(emailInput.value).toBe("alice@example.com");
    expect(passwordInput.value).toBe("hunter2pass");
  });

  // --- Client-side password-length validation (page-specific — the
  // login page has no equivalent of this "tooShort" state at all) ---

  it("shows no length warning and keeps the submit button enabled before any password is typed", () => {
    renderRegisterPage();
    expect(screen.queryByText(/at least 8 characters/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create account/i })).toBeEnabled();
  });

  it("shows an inline warning and disables submit once a too-short password is typed", async () => {
    const user = userEvent.setup();
    renderRegisterPage();

    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "short");

    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create account/i })).toBeDisabled();
    expect(registerAccount).not.toHaveBeenCalled();
  });

  it("clears the warning and re-enables submit once the password reaches 8 characters", async () => {
    const user = userEvent.setup();
    renderRegisterPage();

    const passwordInput = screen.getByLabelText("Password");
    await user.type(passwordInput, "short");
    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument();

    await user.type(passwordInput, "er"); // "shorter" -> 7 chars, still short
    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument();

    await user.type(passwordInput, "1"); // 8 chars now
    expect(screen.queryByText("At least 8 characters.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create account/i })).toBeEnabled();
  });

  it("submitting a too-short password does not call the API even via form submit", async () => {
    const user = userEvent.setup();
    renderRegisterPage();

    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    // Query the password input BEFORE typing a too-short value — once
    // "short" triggers the length warning, the warning text renders
    // INSIDE the same <label> as the input (see app/register/page.tsx),
    // which changes the label's accessible name to "Password At least 8
    // characters." — an exact getByLabelText("Password") lookup after
    // that point would fail even though the input itself hasn't moved.
    // Capturing the element reference once, up front, sidesteps that
    // without weakening what this test actually checks.
    const passwordInput = screen.getByLabelText("Password");
    await user.type(passwordInput, "short");
    // The button is disabled (covered above); this additionally proves
    // the page's own submit() guard (`if (busy || tooShort) return;`)
    // holds even if a disabled button were somehow still triggered
    // (e.g. pressing Enter in a form field submits the form directly,
    // bypassing the button's disabled state).
    await user.type(passwordInput, "{Enter}");

    expect(registerAccount).not.toHaveBeenCalled();
  });

  // --- Submission ---

  it("submitting a valid email and password calls registerAccount and redirects to /library", async () => {
    const user = userEvent.setup();
    vi.mocked(registerAccount).mockResolvedValueOnce({
      access_token: "new-token",
      token_type: "bearer",
      expires_in_minutes: 60,
    });
    vi.mocked(getMe).mockResolvedValueOnce({ id: 1, email: "alice@example.com", is_active: true, created_at: null });

    renderRegisterPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => expect(registerAccount).toHaveBeenCalledWith("alice@example.com", "hunter2pass"));
    // Unlike LoginPage, RegisterPage has no ?redirect= support at all —
    // it always sends a newly-created account to /library.
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/library"));
  });

  it("shows the backend's error message on a 409 (email already registered), and does not redirect", async () => {
    const user = userEvent.setup();
    // A real ApiError instance -- the page's catch block branches on
    // `err instanceof ApiError` (see app/register/page.tsx), so a
    // plain-object stand-in with matching fields would silently take
    // the wrong branch and this test would pass for the wrong reason.
    vi.mocked(registerAccount).mockRejectedValueOnce(new ApiError("An account with this email already exists.", 409));

    renderRegisterPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    expect(await screen.findByText("An account with this email already exists.")).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("falls back to a generic message when the thrown error isn't an ApiError", async () => {
    const user = userEvent.setup();
    vi.mocked(registerAccount).mockRejectedValueOnce(new TypeError("Failed to fetch"));

    renderRegisterPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    // Deliberately a DIFFERENT generic message than LoginPage's "Could
    // not sign in. Try again." — proves this page's own fallback string
    // is under test, not LoginPage's copy-pasted into a shared helper.
    expect(await screen.findByText("Could not create an account. Try again.")).toBeInTheDocument();
  });

  it("shows a submitting state while the request is in flight and disables the submit button", async () => {
    const user = userEvent.setup();
    let resolveRegister!: (value: { access_token: string; token_type: "bearer"; expires_in_minutes: number }) => void;
    vi.mocked(registerAccount).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveRegister = resolve;
      }),
    );

    renderRegisterPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    const submitButton = await screen.findByRole("button", { name: /creating account/i });
    expect(submitButton).toBeDisabled();

    resolveRegister({ access_token: "t", token_type: "bearer", expires_in_minutes: 60 });
    vi.mocked(getMe).mockResolvedValueOnce({ id: 1, email: "alice@example.com", is_active: true, created_at: null });

    await waitFor(() => expect(pushMock).toHaveBeenCalled());
  });

  it("clears a previous error message on a successful resubmit", async () => {
    const user = userEvent.setup();
    vi.mocked(registerAccount).mockRejectedValueOnce(new ApiError("An account with this email already exists.", 409));

    renderRegisterPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /create account/i }));
    expect(await screen.findByText("An account with this email already exists.")).toBeInTheDocument();

    vi.mocked(registerAccount).mockResolvedValueOnce({
      access_token: "new-token",
      token_type: "bearer",
      expires_in_minutes: 60,
    });
    vi.mocked(getMe).mockResolvedValueOnce({ id: 2, email: "alice@example.com", is_active: true, created_at: null });

    await user.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/library"));
    expect(screen.queryByText("An account with this email already exists.")).not.toBeInTheDocument();
  });

  it("links to the login page", () => {
    renderRegisterPage();
    const loginLink = screen.getByRole("link", { name: /sign in/i });
    expect(loginLink).toHaveAttribute("href", "/login");
  });
});