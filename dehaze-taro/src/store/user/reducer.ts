import type {UserInfo} from "dehaze-sdk-js";
import {UserState} from ".";
import {RESET_USER_INFO, SET_USER_INFO} from "./actions";

interface SetUserInfoAction {
  type: typeof SET_USER_INFO;
  payload: UserInfo;
}

interface ResetUserInfoAction {
  type: typeof RESET_USER_INFO;
}

export type UserActionTypes = SetUserInfoAction | ResetUserInfoAction;

// Initial State
const INITIAL_USER_STATE: UserInfo = {
  roles: [],
  perms: [],
};

const initialState: UserState = {
  user: INITIAL_USER_STATE,
};

// Reducer
export default function userReducer(
  state = initialState,
  action: UserActionTypes
): UserState {
  switch (action.type) {
    case SET_USER_INFO:
      return {
        ...state,
        user: action.payload,
      };
    case RESET_USER_INFO:
      return {
        ...state,
        user: INITIAL_USER_STATE,
      };
    default:
      return state;
  }
}
