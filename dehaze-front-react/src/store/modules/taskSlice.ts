import {
  TaskAPI,
  type TaskVO,
  type TaskQuery,
  type TaskStatus,
} from "dehaze-sdk-js";
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

/** 终态状态集合（无需继续轮询） */
export const TERMINAL_STATUSES: TaskStatus[] = [
  "COMPLETED",
  "FAILED",
  "CANCELLED",
];

/** 需要轮询的状态集合 */
export const POLLING_STATUSES: TaskStatus[] = ["PENDING", "PROCESSING"];

interface TaskState {
  /** 任务列表 */
  taskList: TaskVO[];
  /** 任务总数 */
  total: number;
  /** 列表加载状态 */
  loading: boolean;
  /** 当前查看的任务详情 */
  currentTask: TaskVO | null;
  /** 轮询定时器ID */
  pollingTimer: number | null;
}

const initialState: TaskState = {
  taskList: [],
  total: 0,
  loading: false,
  currentTask: null,
  pollingTimer: null,
};

/** 分页查询任务列表 */
export const fetchTaskList = createAsyncThunk(
  "task/fetchList",
  async (queryParams: TaskQuery) => {
    const response = await TaskAPI.getPage(queryParams);
    return response;
  }
);

/** 查询单个任务状态 */
export const fetchTaskStatus = createAsyncThunk(
  "task/fetchStatus",
  async (taskId: string) => {
    const response = await TaskAPI.getStatus(taskId);
    return response;
  }
);

/** 取消任务 */
export const cancelTask = createAsyncThunk(
  "task/cancel",
  async (taskId: string) => {
    await TaskAPI.cancel(taskId);
    return taskId;
  }
);

const taskSlice = createSlice({
  name: "task",
  initialState,
  reducers: {
    /** 设置当前查看的任务 */
    setCurrentTask: (state, action: { payload: TaskVO | null }) => {
      state.currentTask = action.payload;
    },
    /** 设置轮询定时器ID */
    setPollingTimer: (state, action: { payload: number | null }) => {
      state.pollingTimer = action.payload;
    },
    /** 清除轮询定时器 */
    clearPollingTimer: (state) => {
      state.pollingTimer = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchTaskList.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchTaskList.fulfilled, (state, action) => {
        state.taskList = action.payload.list || [];
        state.total = action.payload.total || 0;
        state.loading = false;
      })
      .addCase(fetchTaskList.rejected, (state) => {
        state.loading = false;
      })
      .addCase(fetchTaskStatus.fulfilled, (state, action) => {
        const task = action.payload;
        const index = state.taskList.findIndex((t) => t.taskId === task.taskId);
        if (index !== -1) {
          state.taskList[index] = task;
        }
        // 同步更新当前详情中的任务
        if (state.currentTask?.taskId === task.taskId) {
          state.currentTask = task;
        }
      })
      .addCase(cancelTask.fulfilled, (state, action) => {
        const taskId = action.payload;
        const index = state.taskList.findIndex((t) => t.taskId === taskId);
        if (index !== -1) {
          state.taskList[index] = {
            ...state.taskList[index],
            status: "CANCELLED",
            completedAt: new Date().toISOString(),
          };
        }
        if (state.currentTask?.taskId === taskId) {
          state.currentTask = {
            ...state.currentTask,
            status: "CANCELLED",
            completedAt: new Date().toISOString(),
          };
        }
      });
  },
});

export const { setCurrentTask, setPollingTimer, clearPollingTimer } =
  taskSlice.actions;
export default taskSlice.reducer;
