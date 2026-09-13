import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  clearStoredToken,
  getStoredToken,
  notifyUnauthorized,
  onUnauthorized,
  setStoredToken,
} from "@/lib/auth-storage";

describe("auth-storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("returns null when nothing is stored", () => {
    expect(getStoredToken()).toBeNull();
  });

  it("round-trips a token through set/get", () => {
    setStoredToken("abc.def.ghi");
    expect(getStoredToken()).toBe("abc.def.ghi");
  });

  it("clearStoredToken removes a stored token", () => {
    setStoredToken("abc.def.ghi");
    clearStoredToken();
    expect(getStoredToken()).toBeNull();
  });

  it("getStoredToken does not throw when localStorage access fails", () => {
    const spy = vi.spyOn(window.localStorage.__proto__, "getItem").mockImplementation(() => {
      throw new Error("storage disabled");
    });
    expect(getStoredToken()).toBeNull();
    spy.mockRestore();
  });

  it("notifyUnauthorized clears the stored token", () => {
    setStoredToken("abc.def.ghi");
    notifyUnauthorized();
    expect(getStoredToken()).toBeNull();
  });

  it("notifyUnauthorized calls every subscribed listener", () => {
    const listenerA = vi.fn();
    const listenerB = vi.fn();
    const unsubA = onUnauthorized(listenerA);
    const unsubB = onUnauthorized(listenerB);

    notifyUnauthorized();

    expect(listenerA).toHaveBeenCalledTimes(1);
    expect(listenerB).toHaveBeenCalledTimes(1);
    unsubA();
    unsubB();
  });

  it("onUnauthorized's unsubscribe function stops future notifications", () => {
    const listener = vi.fn();
    const unsubscribe = onUnauthorized(listener);
    unsubscribe();

    notifyUnauthorized();

    expect(listener).not.toHaveBeenCalled();
  });

  it("a listener added twice via separate calls both fire independently", () => {
    const listener = vi.fn();
    const unsub1 = onUnauthorized(listener);
    notifyUnauthorized();
    expect(listener).toHaveBeenCalledTimes(1);
    unsub1();

    const unsub2 = onUnauthorized(listener);
    notifyUnauthorized();
    expect(listener).toHaveBeenCalledTimes(2);
    unsub2();
  });
});