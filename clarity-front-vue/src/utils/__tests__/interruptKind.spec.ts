import { describe, expect, it } from "vitest";
import type { ConfirmKind, Interrupt } from "clarity-sdk-js";
import {
  KNOWN_CONFIRM_KINDS,
  confirmKindStatus,
  isConfirmKindInvalid,
  isKnownConfirmKind,
  resolveConfirmKind,
} from "../interruptKind";

const interrupt = (over: Partial<Interrupt> = {}): Interrupt => ({
  id: "i1",
  type: "confirm",
  createdAt: "2026-01-01T00:00:00Z",
  expiresAt: "2099-01-01T00:00:00Z",
  actionable: true,
  ...over,
});

describe("interruptKind", () => {
  it("合法子类型列表与契约 ConfirmKind 一致", () => {
    expect([...KNOWN_CONFIRM_KINDS].sort()).toEqual(
      [
        "algorithm_recommend",
        "dangerous_op",
        "tool_permission",
        "write_conflict",
      ].sort()
    );
  });

  describe("isKnownConfirmKind", () => {
    it.each([...KNOWN_CONFIRM_KINDS])("%s 合法", (kind) => {
      expect(isKnownConfirmKind(kind)).toBe(true);
    });

    it.each([["bogus"], [""], [null], [undefined], [1], [{}]])(
      "契约外取值 %s 不合法",
      (raw) => {
        expect(isKnownConfirmKind(raw)).toBe(false);
      }
    );
  });

  describe("confirmKindStatus", () => {
    it.each([...KNOWN_CONFIRM_KINDS])("%s 为 ok", (kind) => {
      expect(confirmKindStatus(interrupt({ confirmKind: kind }))).toBe("ok");
    });

    it("缺失（null / undefined）为 missing", () => {
      expect(confirmKindStatus(interrupt({ confirmKind: null }))).toBe(
        "missing"
      );
      expect(confirmKindStatus(interrupt({}))).toBe("missing");
    });

    it("契约外取值为 unknown", () => {
      expect(
        confirmKindStatus(
          interrupt({ confirmKind: "bogus" as Interrupt["confirmKind"] })
        )
      ).toBe("unknown");
    });

    it("非 confirm 中断为 not_confirm（即使携带 confirmKind）", () => {
      expect(
        confirmKindStatus(
          interrupt({ type: "quota", confirmKind: "dangerous_op" })
        )
      ).toBe("not_confirm");
    });
  });

  describe("isConfirmKindInvalid", () => {
    it("missing / unknown 视为不可用", () => {
      expect(isConfirmKindInvalid(interrupt({}))).toBe(true);
      expect(
        isConfirmKindInvalid(
          interrupt({ confirmKind: "bogus" as Interrupt["confirmKind"] })
        )
      ).toBe(true);
    });

    it("合法子类型与非 confirm 中断都不算不可用", () => {
      expect(
        isConfirmKindInvalid(interrupt({ confirmKind: "dangerous_op" }))
      ).toBe(false);
      expect(isConfirmKindInvalid(interrupt({ type: "quota" }))).toBe(false);
    });
  });

  describe("resolveConfirmKind", () => {
    it.each([...KNOWN_CONFIRM_KINDS])("%s 原样返回", (kind) => {
      expect(resolveConfirmKind(interrupt({ confirmKind: kind }))).toBe(kind);
    });

    it("未知 / 缺失 / 非 confirm 一律 undefined", () => {
      expect(
        resolveConfirmKind(
          interrupt({ confirmKind: "bogus" as Interrupt["confirmKind"] })
        )
      ).toBeUndefined();
      expect(resolveConfirmKind(interrupt({}))).toBeUndefined();
      expect(resolveConfirmKind(interrupt({ type: "quota" }))).toBeUndefined();
    });

    it("只认 confirmKind，不从 action 推导", () => {
      expect(
        resolveConfirmKind(
          interrupt({ confirmKind: "dangerous_op", action: "write_conflict" })
        )
      ).toBe<ConfirmKind>("dangerous_op");
    });
  });
});
