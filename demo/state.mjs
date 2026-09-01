export function createDemoState(exercise, requestToken = 0) {
  return {
    selectedExerciseId: exercise.id,
    baseSnapshot: { ...exercise.baseSnapshot },
    fineTunedMessages: starterMessages(exercise),
    activeTab: 'fine-tuned',
    pending: false,
    error: null,
    requestToken,
  };
}

function starterMessages(exercise) {
  return [
    { role: 'user', content: exercise.starter.user },
    { role: 'assistant', content: exercise.starter.assistant },
  ];
}

export function reduceDemoState(state, action) {
  switch (action.type) {
    case 'select-exercise':
      return createDemoState(action.exercise, state.requestToken + 1);
    case 'restart':
      return createDemoState(action.exercise, state.requestToken + 1);
    case 'select-tab':
      return action.tab === 'base' || action.tab === 'fine-tuned'
        ? { ...state, activeTab: action.tab }
        : state;
    case 'submit': {
      const requestToken = state.requestToken + 1;
      return {
        ...state,
        fineTunedMessages: [
          ...state.fineTunedMessages,
          { role: 'user', content: action.content },
        ],
        pending: true,
        error: null,
        requestToken,
      };
    }
    case 'retry':
      return {
        ...state,
        pending: true,
        error: null,
        requestToken: state.requestToken + 1,
      };
    case 'response':
      if (action.token !== state.requestToken) return state;
      return {
        ...state,
        fineTunedMessages: [
          ...state.fineTunedMessages,
          { role: 'assistant', content: action.content },
        ],
        pending: false,
        error: null,
      };
    case 'failure':
      if (action.token !== state.requestToken) return state;
      return { ...state, pending: false, error: action.message };
    default:
      return state;
  }
}
