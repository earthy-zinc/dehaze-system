import { describe, it, expect, vi, beforeEach } from "vitest";
import { configureStore } from "@reduxjs/toolkit";
import favoriteReducer, {
  toggleFavorite,
  setStatus,
} from "@/store/modules/favoriteSlice";
import type { DisPatchType } from "@/store/index";
import { FavoriteAPI } from "dehaze-sdk-js";

vi.mock("dehaze-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("dehaze-sdk-js")>();
  return {
    ...actual,
    FavoriteAPI: {
      add: vi.fn(),
      getPage: vi.fn(),
      deleteByIds: vi.fn(),
      getStatus: vi.fn(),
      getCount: vi.fn(),
    },
  };
});

const mockApi = vi.mocked(FavoriteAPI);

function createStore() {
  const store = configureStore({
    reducer: { favorite: favoriteReducer },
  });
  return { store, dispatch: store.dispatch as DisPatchType };
}

describe("favoriteSlice toggleFavorite", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("未收藏时 toggle 调用 add 并将状态置为 true", async () => {
    mockApi.add.mockResolvedValue(1 as any);
    const { store, dispatch } = createStore();

    await dispatch(toggleFavorite({ targetType: "algorithm", targetId: 10 }));

    expect(mockApi.add).toHaveBeenCalledWith({
      targetType: "algorithm",
      targetId: 10,
    });
    expect(mockApi.deleteByIds).not.toHaveBeenCalled();
    expect(store.getState().favorite.status.algorithm[10]).toBe(true);
  });

  it("已收藏时 toggle 通过 getPage 找到记录 id 后 deleteByIds 并将状态置为 false", async () => {
    mockApi.getPage.mockResolvedValue({
      total: 1,
      list: [{ id: 99, targetType: "algorithm", targetId: 10 } as any],
    });
    mockApi.deleteByIds.mockResolvedValue(undefined);
    const { store, dispatch } = createStore();
    dispatch(
      setStatus({ targetType: "algorithm", targetId: 10, favorited: true })
    );

    await dispatch(toggleFavorite({ targetType: "algorithm", targetId: 10 }));

    expect(mockApi.getPage).toHaveBeenCalledWith({
      targetType: "algorithm",
      pageSize: 500,
    });
    expect(mockApi.deleteByIds).toHaveBeenCalledWith([99]);
    expect(mockApi.add).not.toHaveBeenCalled();
    expect(store.getState().favorite.status.algorithm[10]).toBe(false);
  });
});
