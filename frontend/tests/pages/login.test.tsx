import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import LoginPage from "@/app/login/page";
import { AuthProvider } from "@/components/auth-provider";

// Same integration shape as tests/components/auth-provider.test.tsx:
// mock the NETWORK boundary (lib/api.ts) and let the real AuthProvider
// run, so this suite exercises the actual page + provider wiring a user
// experiences, not a re-implementation of AuthProvider's own logic
// (already covered by that file).
vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getMe: vi.fn(),
    login: vi.fn(),
    registerAccount: vi.fn(),
  };
});

import { getMe, login as apiLogin, ApiError } from "@/lib/api";

const pushMock = vi.fn();
let searchParamsValue = "";
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
  useSearchParams: () => new URLSearchParams(searchParamsValue),
}));

function renderLoginPage() {
  return render(
    <AuthProvider>
      <LoginPage />
    </AuthProvider>,
  );
}

describe("LoginPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    pushMock.mockReset();
    searchParamsValue = "";
    vi.mocked(getMe).mockReset();
    vi.mocked(apiLogin).mockReset();
  });

  it("renders the sign-in form with email and password fields", () => {
    renderLoginPage();
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("marks email and password as required fields", () => {
    renderLoginPage();
    expect(screen.getByLabelText("Email")).toBeRequired();
    expect(screen.getByLabelText("Password")).toBeRequired();
  });

  it("uses the correct input types for email and password", () => {
    renderLoginPage();
    expect(screen.getByLabelText("Email")).toHaveAttribute("type", "email");
    expect(screen.getByLabelText("Password")).toHaveAttribute("type", "password");
  });

  it("lets the user type into the email and password fields", async () => {
    const user = userEvent.setup();
    renderLoginPage();

    const emailInput = screen.getByLabelText("Email") as HTMLInputElement;
    const passwordInput = screen.getByLabelText("Password") as HTMLInputElement;

    await user.type(emailInput, "alice@example.com");
    await user.type(passwordInput, "hunter2pass");

    expect(emailInput.value).toBe("alice@example.com");
    expect(passwordInput.value).toBe("hunter2pass");
  });

  it("submitting valid credentials calls the login API and redirects to /library by default", async () => {
    const user = userEvent.setup();
    vi.mocked(apiLogin).mockResolvedValueOnce({
      access_token: "new-token",
      token_type: "bearer",
      expires_in_minutes: 60,
    });
    vi.mocked(getMe).mockResolvedValueOnce({ id: 1, email: "alice@example.com", is_active: true, created_at: null });

    renderLoginPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => expect(apiLogin).toHaveBeenCalledWith("alice@example.com", "hunter2pass"));
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/library"));
  });

  it("redirects to the ?redirect= target instead of /library when present", async () => {
    const user = userEvent.setup();
    searchParamsValue = "redirect=%2Fupload";
    vi.mocked(apiLogin).mockResolvedValueOnce({
      access_token: "new-token",
      token_type: "bearer",
      expires_in_minutes: 60,
    });
    vi.mocked(getMe).mockResolvedValueOnce({ id: 1, email: "alice@example.com", is_active: true, created_at: null });

    renderLoginPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/upload"));
  });

  it("shows the backend's error message on failed authentication, and does not redirect", async () => {
    const user = userEvent.setup();
    // A real ApiError instance -- the page's catch block branches on
    // `err instanceof ApiError` (see app/login/page.tsx), so a
    // plain-object stand-in with matching fields would silently take
    // the wrong branch and this test would pass for the wrong reason.
    vi.mocked(apiLogin).mockRejectedValueOnce(new ApiError("Incorrect email or password.", 401));

    renderLoginPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "wrong-password");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText("Incorrect email or password.")).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("falls back to a generic message when the thrown error isn't an ApiError", async () => {
    const user = userEvent.setup();
    vi.mocked(apiLogin).mockRejectedValueOnce(new TypeError("Failed to fetch"));

    renderLoginPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText("Could not sign in. Try again.")).toBeInTheDocument();
  });

  it("shows a submitting state while the request is in flight and disables the submit button", async () => {
    const user = userEvent.setup();
    let resolveLogin!: (value: { access_token: string; token_type: "bearer"; expires_in_minutes: number }) => void;
    vi.mocked(apiLogin).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveLogin = resolve;
      }),
    );

    renderLoginPage();
    await user.type(screen.getByLabelText("Email"), "alice@example.com");
    await user.type(screen.getByLabelText("Password"), "hunter2pass");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    const submitButton = await screen.findByRole("button", { name: /signing in/i });
    expect(submitButton).toBeDisabled();

    resolveLogin({ access_token: "t", token_type: "bearer", expires_in_minutes: 60 });
    vi.mocked(getMe).mockResolvedValueOnce({ id: 1, email: "alice@example.com", is_active: true, created_at: null });

    await waitFor(() => expect(pushMock).toHaveBeenCalled());
  });

  it("links to the register page", () => {
    renderLoginPage();
    const registerLink = screen.getByRole("link", { name: /register/i });
    expect(registerLink).toHaveAttribute("href", "/register");
  });
});