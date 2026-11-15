// Once the server is warm a search answers in a few milliseconds, too fast to see what
// was done. So the answer is held back until every loading step has had its turn on
// screen. Set STEP_MS to 0 to show results the moment they arrive.
export const STEP_MS = 650;

// How long a loading state with this many steps stays up at the least.
export const holdFor = (steps) => new Promise((resolve) => {
  setTimeout(resolve, steps * STEP_MS);
});
