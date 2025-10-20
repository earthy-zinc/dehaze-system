import { combineReducers } from "redux";
import login from "./login";
import user from "../store/user/reducer";

export default combineReducers({
  login,
  user,
});
