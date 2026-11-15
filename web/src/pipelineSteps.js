// The lines a loading state shows: what the server does for this request, in order.
// They follow recsys/service.py, so change them when the pipeline changes.
// Each step is its words and the drawing that goes with it (see StepIcon.jsx).

const step = (icon, text) => ({ icon, text });

const catalogue = (recipes) =>
  typeof recipes === 'number' && recipes > 0 ? `${recipes.toLocaleString('en-US')} recipes` : 'every recipe';

const count = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

const BY_MODEL = {
  learned: () => [
    step('retrieve', 'Pulling candidates by TF-IDF, overlap and ingredient embeddings'),
    step('trees', 'Scoring each candidate with the boosted-tree ranker'),
  ],
  embedding: (recipes) => [
    step('embed', 'Looking up their ingredient embeddings'),
    step('compare', `Comparing them with ${catalogue(recipes)}`),
  ],
  overlap: (recipes) => [step('overlap', `Counting shared ingredients across ${catalogue(recipes)}`)],
  tfidf: (recipes) => [
    step('weigh', 'Weighing rare ingredients above common ones'),
    step('compare', `Comparing them with ${catalogue(recipes)}`),
  ],
};

export function searchSteps(run, recipes) {
  if (!run) return [step('compare', 'Finding recipes')];
  if (run.mode === 'describe') {
    return [
      step('read', 'Reading your description'),
      step('embed', 'Encoding it with the MiniLM text model'),
      step('compare', `Comparing it with ${catalogue(recipes)}`),
      step('sort', 'Sorting by closeness'),
    ];
  }
  const ingredients = run.ingredients || [];
  if (!ingredients.length) {
    return [
      step('read', 'Reading your filters'),
      step('compare', `Scanning ${catalogue(recipes)}`),
      step('sort', 'Sorting the matches'),
    ];
  }
  const middle = (BY_MODEL[run.model] || BY_MODEL.learned)(recipes);
  return [
    step('match', `Matching your ${count(ingredients.length, 'ingredient')} to the ones the models know`),
    ...middle,
    step('sort', 'Adding your filters and likes, then sorting'),
  ];
}

export const similarSteps = (recipes) => [
  step('embed', 'Reading the ingredient and text vectors of this recipe'),
  step('compare', `Finding its nearest neighbours among ${catalogue(recipes)}`),
];

export const recipeSteps = () => [
  step('read', 'Opening the recipe'),
  step('palette', 'Reading the ingredient colours for its cover'),
];
