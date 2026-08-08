/**
 * 消息未读数 zustand store
 *
 * 供 TabBar 角标与各页面订阅未读消息数。
 */
import { create } from 'zustand';

interface MessagesState {
  unreadCount: number;
  setUnreadCount: (count: number) => void;
  decrementUnread: (delta?: number) => void;
}

export const useMessagesStore = create<MessagesState>((set) => ({
  unreadCount: 0,
  setUnreadCount: (count: number) => set({ unreadCount: count }),
  decrementUnread: (delta = 1) =>
    set((state) => ({ unreadCount: Math.max(0, state.unreadCount - delta) }),
  ),
}));
