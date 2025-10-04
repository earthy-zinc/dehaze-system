const INITIAL_STATE = {
  num: 0,
};

export default function counter(state = INITIAL_STATE, action) {
  switch (action.type) {
    case 1:
      return {
        ...state,
        num: state.num + 1,
      };
    case 2:
      return {
        ...state,
        num: state.num - 1,
      };
    default:
      return state;
  }
}
