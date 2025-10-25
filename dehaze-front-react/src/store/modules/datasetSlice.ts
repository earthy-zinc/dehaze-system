import { DatasetAPI, Dataset, DatasetQuery } from "dehaze-sdk-js";
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";

// Define the initial state
interface DatasetState {
  datasetList: Dataset[];
  loading: boolean;
}

const initialState: DatasetState = {
  datasetList: [],
  loading: false,
};

// Async thunk for fetching dataset list
export const getDatasetList = createAsyncThunk(
  "dataset/getList",
  async (queryParams?: DatasetQuery) => {
    const response = await DatasetAPI.getList(queryParams);
    return response;
  }
);

// Async thunk for adding a dataset
export const addDataset = createAsyncThunk(
  "dataset/add",
  async (data: Dataset) => {
    const response = await DatasetAPI.add(data);
    return response;
  }
);

// Async thunk for updating a dataset
export const updateDataset = createAsyncThunk(
  "dataset/update",
  async ({ id, data }: { id: number; data: Dataset }) => {
    const response = await DatasetAPI.update(id, data);
    return response;
  }
);

// Async thunk for deleting datasets
export const deleteDatasetByIds = createAsyncThunk(
  "dataset/delete",
  async (ids: string[]) => {
    const response = await DatasetAPI.deleteByIds(ids);
    return response;
  }
);

// Create the slice
const datasetSlice = createSlice({
  name: "dataset",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(getDatasetList.pending, (state) => {
        state.loading = true;
      })
      .addCase(getDatasetList.fulfilled, (state, action) => {
        state.datasetList = action.payload;
        state.loading = false;
      })
      .addCase(getDatasetList.rejected, (state) => {
        state.loading = false;
      });
  },
});

// Persist configuration
const datasetPersistConfig = {
  key: "dataset",
  storage,
  whitelist: ["datasetList"],
};

export default persistReducer(datasetPersistConfig, datasetSlice.reducer);
