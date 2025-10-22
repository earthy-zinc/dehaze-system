import {
  legacy_createStore as createStore,
  applyMiddleware,
  compose,
  Middleware,
} from "redux";
import thunkMiddleware from "redux-thunk";
import logger from "redux-logger";
import { combineReducers } from "redux";
import user from "@/store/user/reducer";

const composeEnhancers =
  typeof window === "object" && window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__
    ? window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__({
        // Specify extension's options like name, actionsBlacklist, actionsCreators, serialize...
      })
    : compose;

const middlewares = [thunkMiddleware] as Middleware[];

if (process.env.NODE_ENV === "development") {
  middlewares.push(logger as Middleware);
}

const enhancer = composeEnhancers(
  applyMiddleware(...middlewares)
  // other store enhancers if any
);

const rootReducer = combineReducers({
  user,
});

export default function configStore() {
  const store = createStore(rootReducer, enhancer);
  return store;
}
