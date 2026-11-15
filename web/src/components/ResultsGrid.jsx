/* eslint-disable react/prop-types */
import RecipeCard from './RecipeCard';
import { ArrowRight } from 'lucide-react';
import Pagination from './Pagination';
import TimingCard from './TimingCard';
import Pipeline from './Pipeline';
import { searchSteps } from '../pipelineSteps';

// What the current results were asked for, as plain lines.
const FilterSummary = ({ run }) => {
  const rows = [];
  if (run.mode === 'describe') {
    rows.push(['Search', `\u201c${run.query}\u201d`]);
  } else {
    if (run.ingredients.length) rows.push(['Ingredients', run.ingredients.join(', ')]);
    if (run.cuisine.length) rows.push(['Cuisine', run.cuisine.join(', ')]);
    if (run.course) rows.push(['Course', run.course]);
  }
  if (run.veg) rows.push(['Diet', 'Vegetarian only']);
  if (!rows.length) return null;
  return (
    <dl className="mt-3 space-y-1 text-[15px] leading-relaxed">
      {rows.map(([label, value]) => (
        <div key={label} className="flex gap-3">
          <dt className="w-24 shrink-0 text-slate-500">{label}</dt>
          <dd className="min-w-0 text-slate-800">{value}</dd>
        </div>
      ))}
    </dl>
  );
};

const EXAMPLES = [
  'something creamy and spicy with spinach',
  'a quick breakfast with oats and banana',
  'light lentil soup for a cold evening',
  'crispy snack to have with tea',
];

const Message = ({ title, children }) => (
  <div className="max-w-md py-12">
    <h2 className="font-serif text-2xl font-medium text-ink">{title}</h2>
    <div className="mt-2 text-base leading-relaxed text-slate-600">{children}</div>
  </div>
);

const ResultsGrid = ({
  status, error, items, run, recipes, timing, page, pageSize, onPage, liked, onToggleLike, onRetry, onExample,
}) => {
  if (status === 'loading') {
    // The search as asked, then the steps the server is working through, over the grid the results will fill.
    return (
      <div aria-busy="true">
        <header className="pb-6">
          <h1 className="font-serif text-[28px] font-medium tracking-[-0.01em] text-ink lg:text-[32px]">
            Finding your matches
          </h1>
          {run && <FilterSummary run={run} />}
          <Pipeline className="mt-4" label="Finding recipes" steps={searchSteps(run, recipes)} />
        </header>
        <div aria-hidden="true" className="grid grid-cols-1 gap-x-8 gap-y-10 border-t border-slate-200 pt-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }, (_, i) => (
            <div key={i} className="animate-pulse motion-reduce:animate-none" style={{ animationDelay: `${i * 140}ms` }}>
              <div className="aspect-[3/2] rounded-[3px] bg-[#efede8]" />
              <div className="mt-5 h-4 w-32 rounded bg-[#efede8]" />
              <div className="mt-4 h-5 w-4/5 rounded bg-[#efede8]" />
              <div className="mt-4 h-4 w-16 rounded bg-[#efede8]" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <Message title="Something went wrong">
        <p className="text-red-700" role="alert">{error}</p>
        <button type="button" onClick={onRetry} className="btn-quiet mt-5">Try again</button>
      </Message>
    );
  }

  if (status === 'idle') {
    return (
      <div>
        <h2 className="text-sm font-medium text-ink">Or start from a description</h2>
        <ul className="mt-3 border-t border-slate-200 sm:grid sm:grid-cols-2 sm:gap-x-10">
          {EXAMPLES.map((text) => (
            <li key={text} className="border-b border-slate-200">
              <button
                type="button"
                onClick={() => onExample(text)}
                className="group flex min-h-14 w-full items-center justify-between gap-4 py-3 text-left font-serif text-lg italic text-slate-700 transition-colors hover:text-ink"
              >
                {text}
                <ArrowRight size={18} aria-hidden="true" className="shrink-0 text-slate-400 transition-transform group-hover:translate-x-0.5 group-hover:text-ink" />
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  const start = (page - 1) * pageSize;
  return (
    <div>
      <header className="pb-6">
        <h1 className="font-serif text-[28px] font-medium tracking-[-0.01em] text-ink lg:text-[32px]">
          {items.length ? 'Top matches' : 'No matches'}
        </h1>
        {run && <FilterSummary run={run} />}
        <div className="mt-4">
          <TimingCard count={items.length} current={timing.current} previous={timing.previous} />
        </div>
      </header>

      {items.length === 0 ? (
        <div className="border-t border-slate-200">
          <Message title="No recipes matched">
            Try fewer filters, different ingredients, or another model.
          </Message>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-x-8 gap-y-10 border-t border-slate-200 pt-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {items.slice(start, start + pageSize).map((recipe) => (
              <RecipeCard
                key={recipe.id}
                recipe={recipe}
                liked={liked.includes(recipe.id)}
                onToggleLike={() => onToggleLike(recipe.id)}
              />
            ))}
          </div>
          <div className="mt-10 border-t border-slate-200 pt-6">
            <Pagination total={items.length} pageSize={pageSize} currentPage={page} onPage={onPage} />
          </div>
        </>
      )}
    </div>
  );
};

export default ResultsGrid;
