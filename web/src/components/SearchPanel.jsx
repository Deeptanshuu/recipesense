/* eslint-disable react/prop-types */
import { useState, useMemo, useEffect, useRef, useId } from 'react';
import { X, Plus, Search, ChevronDown } from 'lucide-react';
import VegToggle from './VegToggle';
import ModelPicker from './ModelPicker';
import ingredientsData from '../ingredients.json';
import { removeEmojis } from '../utils';
import { useStored } from '../searchStore';

const INGREDIENTS = ingredientsData.ingredients.map((name, i) => ({ id: `ing_${i}`, name: removeEmojis(name) }));
const BY_ID = new Map(INGREDIENTS.map((i) => [i.id, i]));
// The list is ordered by how often an ingredient is used. The quick picks are the first
// few that say something about a dish; salt, oil, water and sugar are in nearly all of them.
const EVERYWHERE = new Set(['salt', 'oil', 'water', 'sugar']);
const QUICK = INGREDIENTS.filter((i) => !EVERYWHERE.has(i.name.toLowerCase())).slice(0, 14);
const SUGGESTIONS = 8;

const MODES = [
  { id: 'ingredients', label: 'By ingredients' },
  { id: 'describe', label: 'Describe it' },
];

// A filter that opens a small panel under its button. The panel closes on Escape and on a
// press outside. `children` may be a function that is handed `close`.
const Filter = ({ label, value, children }) => {
  const [open, setOpen] = useState(false);
  const box = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (box.current && !box.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    // On a phone the panel spans the whole filter row; from sm up it hangs under its own button.
    <div ref={box} className="sm:relative">
      <button
        type="button"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={`chip ${open ? 'border-slate-400 text-ink' : ''}`}
      >
        {label}
        {value && <span className="max-w-[9rem] truncate font-medium text-ink">{value}</span>}
        <ChevronDown size={15} aria-hidden="true" className={`shrink-0 text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div
          role="dialog"
          aria-label={label}
          className="absolute left-3 right-3 top-full z-30 mt-1 rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_16px_40px_-16px_rgb(30_41_59/0.35)] sm:left-0 sm:right-auto sm:mt-2 sm:w-[22rem]"
        >
          {typeof children === 'function' ? children(() => setOpen(false)) : children}
        </div>
      )}
    </div>
  );
};

// The search box: what to look for on top, the filters and the button along the bottom.
// `compact` is its shape once there are results under it.
const SearchPanel = ({
  meta,
  busy,
  hasResults,
  likedCount,
  onClearLikes,
  onSearch,
  onClear,
  className = '',
  preset,
  compact = false,
}) => {
  const [mode, setMode] = useStored('mode', 'ingredients');
  const [model, setModel] = useStored('model', 'learned');
  const [isVeg, setIsVeg] = useStored('veg', false);
  const [query, setQuery] = useStored('query', '');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIngredients, setSelectedIngredients] = useStored('ingredients', []);
  const [customIngredients, setCustomIngredients] = useStored('custom', []);
  const [selectedCuisines, setSelectedCuisines] = useStored('cuisines', []);
  const [selectedCourse, setSelectedCourse] = useStored('course', '');
  const [hint, setHint] = useState('');
  const [active, setActive] = useState(0);
  const input = useRef(null);
  const listId = useId();

  useEffect(() => {
    if (!preset) return;
    setMode('describe');
    setQuery(preset.query);
    setHint('');
    onSearch({ mode: 'describe', query: preset.query, veg: isVeg });
    // Runs once per picked example; the other values are read as they are at that moment.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset]);

  const typed = searchTerm.trim();
  const term = typed.toLowerCase();

  const selected = useMemo(
    () => selectedIngredients.map((id) => BY_ID.get(id)).filter(Boolean),
    [selectedIngredients]
  );

  // Names that start with what was typed come first, then names that contain it.
  const matches = useMemo(() => {
    if (!term) return [];
    const rest = INGREDIENTS.filter((i) => !selectedIngredients.includes(i.id));
    const starts = rest.filter((i) => i.name.toLowerCase().startsWith(term));
    const inside = rest.filter((i) => !i.name.toLowerCase().startsWith(term) && i.name.toLowerCase().includes(term));
    return [...starts, ...inside].slice(0, SUGGESTIONS);
  }, [term, selectedIngredients]);

  const canAddCustom =
    typed &&
    !INGREDIENTS.some((i) => i.name.toLowerCase() === term) &&
    !customIngredients.some((c) => c.toLowerCase() === term);

  const addIngredient = (id) => {
    setSelectedIngredients((prev) => (prev.includes(id) ? prev : [...prev, id]));
    setSearchTerm('');
    setHint('');
  };

  const addCustom = () => {
    if (!canAddCustom) return;
    setCustomIngredients([...customIngredients, typed]);
    setSearchTerm('');
    setHint('');
  };

  const options = [
    ...matches.map((i) => ({ key: i.id, label: i.name, pick: () => addIngredient(i.id) })),
    ...(canAddCustom ? [{ key: 'custom', label: `Add “${typed}”`, custom: true, pick: addCustom }] : []),
  ];
  const current = Math.min(active, Math.max(options.length - 1, 0));

  const toggleCuisine = (name) =>
    setSelectedCuisines((prev) => (prev.includes(name) ? prev.filter((c) => c !== name) : [...prev, name]));

  const run = (overrides = {}) => {
    const m = overrides.model || model;
    if (mode === 'describe') {
      if (!query.trim()) {
        setHint('Type what you feel like eating first.');
        return;
      }
      setHint('');
      onSearch({ mode, query: query.trim(), veg: isVeg });
      return;
    }
    const ingredients = [
      ...INGREDIENTS.filter((i) => selectedIngredients.includes(i.id)).map((i) => i.name),
      ...customIngredients,
    ];
    // The API needs something to go on; say so here instead of sending an empty request.
    if (!ingredients.length && !selectedCuisines.length && !selectedCourse && !likedCount) {
      setHint('Pick at least one ingredient, cuisine or course first.');
      return;
    }
    setHint('');
    onSearch({ mode, model: m, ingredients, cuisine: selectedCuisines, course: selectedCourse, veg: isVeg });
  };

  const changeModel = (id) => {
    setModel(id);
    // Re-run so the timing comparison shows up straight away.
    if (hasResults && mode === 'ingredients' && !busy) run({ model: id });
  };

  const clearAll = () => {
    setSelectedIngredients([]);
    setCustomIngredients([]);
    setSelectedCuisines([]);
    setSelectedCourse('');
    setSearchTerm('');
    setQuery('');
    setHint('');
    setIsVeg(false);
    onClear();
  };

  const onInputKey = (e) => {
    if (e.key === 'ArrowDown' && options.length) {
      e.preventDefault();
      setActive(Math.min(current + 1, options.length - 1));
    } else if (e.key === 'ArrowUp' && options.length) {
      e.preventDefault();
      setActive(Math.max(current - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (options.length) options[current].pick();
      else if (!typed && !busy) run();
    } else if (e.key === 'Escape') {
      setSearchTerm('');
    } else if (e.key === 'Backspace' && !searchTerm) {
      // Backspace in an empty box takes back the last pick.
      if (customIngredients.length) setCustomIngredients(customIngredients.slice(0, -1));
      else if (selectedIngredients.length) setSelectedIngredients(selectedIngredients.slice(0, -1));
    }
  };

  const quick = QUICK.filter((i) => !selectedIngredients.includes(i.id)).slice(0, 10);
  const picked = selected.length + customIngredients.length;
  const modelLabel = (meta.models.find((m) => m.id === model) || { label: '' }).label;
  const cuisineValue = selectedCuisines.length === 1 ? selectedCuisines[0] : selectedCuisines.length ? `${selectedCuisines.length} picked` : '';

  return (
    <div className={className}>
      <div className="rounded-[20px] border border-slate-200 bg-white transition-colors focus-within:border-slate-400">
        <div className="flex items-center justify-between gap-3 px-3 pt-3 sm:px-4">
          <div role="tablist" aria-label="Search mode" className="inline-grid grid-cols-2 gap-1 rounded-full bg-[#efede8] p-1">
            {MODES.map((m) => (
              <button
                key={m.id}
                type="button"
                role="tab"
                aria-selected={mode === m.id}
                onClick={() => { setMode(m.id); setHint(''); }}
                className={`min-h-10 whitespace-nowrap rounded-full px-3 text-sm transition-colors sm:px-4 sm:text-[15px] lg:min-h-9 lg:text-sm ${
                  mode === m.id ? 'bg-white font-medium text-ink ring-1 ring-slate-200' : 'text-slate-600 hover:text-ink'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="min-h-10 rounded-full px-3 text-[15px] text-slate-600 transition-colors hover:bg-slate-100 hover:text-red-700 lg:min-h-9 lg:text-sm"
            onClick={clearAll}
          >
            Clear
          </button>
        </div>

        {mode === 'describe' ? (
          <div className="px-4 pb-2 pt-3 sm:px-5">
            <label htmlFor="describe-input" className="sr-only">Describe what you feel like eating</label>
            <textarea
              id="describe-input"
              rows={compact ? 2 : 3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!busy) run();
                }
              }}
              placeholder="Describe what you feel like eating, e.g. something warm and creamy with mushrooms"
              className="block w-full resize-none bg-transparent py-1 text-lg leading-relaxed text-ink placeholder:text-slate-500 focus:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
            />
          </div>
        ) : (
          <>
            <div className="relative px-4 pt-3 sm:px-5">
              {/* A press anywhere in the box puts the cursor in the input. */}
              <div className="flex min-h-12 cursor-text flex-wrap items-center gap-2" onClick={() => input.current && input.current.focus()}>
                {picked === 0 && <Search size={20} aria-hidden="true" className="shrink-0 text-slate-500" />}
                {selected.map((ingredient) => (
                  <button
                    key={ingredient.id}
                    type="button"
                    aria-pressed="true"
                    title="Remove"
                    onClick={(e) => { e.stopPropagation(); setSelectedIngredients(selectedIngredients.filter((x) => x !== ingredient.id)); }}
                    className="chip chip-on"
                  >
                    {ingredient.name}
                    <X size={14} aria-hidden="true" />
                  </button>
                ))}
                {customIngredients.map((name) => (
                  <button
                    key={`custom-${name}`}
                    type="button"
                    aria-pressed="true"
                    title="Remove"
                    onClick={(e) => { e.stopPropagation(); setCustomIngredients(customIngredients.filter((c) => c !== name)); }}
                    className="chip chip-on"
                  >
                    {name}
                    <X size={14} aria-hidden="true" />
                  </button>
                ))}
                <input
                  ref={input}
                  type="text"
                  role="combobox"
                  aria-label="Search ingredients"
                  aria-expanded={options.length > 0}
                  aria-controls={listId}
                  aria-autocomplete="list"
                  aria-activedescendant={options.length ? `${listId}-${current}` : undefined}
                  autoComplete="off"
                  placeholder={picked ? 'Add another' : 'Type an ingredient, like paneer'}
                  value={searchTerm}
                  onChange={(e) => { setSearchTerm(e.target.value); setActive(0); }}
                  onKeyDown={onInputKey}
                  className="min-w-[9rem] flex-1 bg-transparent py-2 text-lg text-ink placeholder:text-slate-500 focus:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
                />
              </div>
              {typed && (
                <ul
                  id={listId}
                  role="listbox"
                  aria-label="Matching ingredients"
                  className="absolute left-3 right-3 top-full z-30 mt-1 overflow-hidden rounded-2xl border border-slate-200 bg-white py-1.5 shadow-[0_16px_40px_-16px_rgb(30_41_59/0.35)] sm:left-4 sm:right-4"
                >
                  {options.map((o, i) => (
                    <li
                      key={o.key}
                      id={`${listId}-${i}`}
                      role="option"
                      aria-selected={i === current}
                      // mousedown, not click, so the input keeps the focus.
                      onMouseDown={(e) => { e.preventDefault(); o.pick(); }}
                      onMouseEnter={() => setActive(i)}
                      className={`flex min-h-11 cursor-pointer items-center gap-2 px-4 text-base ${
                        i === current ? 'bg-[#f3f1ec] text-ink' : 'text-slate-700'
                      }`}
                    >
                      {o.custom && <Plus size={16} aria-hidden="true" className="shrink-0 text-slate-500" />}
                      {o.label}
                    </li>
                  ))}
                  {options.length === 0 && (
                    <li className="px-4 py-2.5 text-[15px] text-slate-500">Already added.</li>
                  )}
                  {matches.length === 0 && canAddCustom && (
                    <li className="px-4 pb-2 pt-1 text-sm text-slate-500">No listed ingredient matches. Press Enter to add it as your own.</li>
                  )}
                </ul>
              )}
            </div>
            {quick.length > 0 && (
              <div className={`flex-wrap items-center gap-2 px-4 pb-1 pt-2 sm:px-5 ${compact ? 'hidden' : 'flex'}`}>
                <p className="mr-1 text-sm text-slate-500">Often used</p>
                {quick.map((ingredient) => (
                  <button
                    key={ingredient.id}
                    type="button"
                    aria-pressed="false"
                    onClick={() => addIngredient(ingredient.id)}
                    className="chip min-h-9 px-3 text-sm lg:min-h-8"
                  >
                    {ingredient.name}
                  </button>
                ))}
              </div>
            )}
          </>
        )}

        <div className="relative mt-3 flex flex-wrap items-center gap-2 border-t border-slate-200 px-3 py-3 sm:px-4">
          <div className="flex items-center gap-2.5 pl-1 pr-2">
            <VegToggle id="veg-toggle" value={isVeg} onChange={setIsVeg} />
            <label htmlFor="veg-toggle" className="cursor-pointer text-[15px] text-slate-700 lg:text-sm">Vegetarian</label>
          </div>
          {mode === 'ingredients' && (
            <>
              <Filter label="Cuisine" value={cuisineValue}>
                <div className="flex flex-wrap gap-2">
                  {meta.cuisines.map((c) => {
                    const on = selectedCuisines.includes(c.name);
                    return (
                      <button key={c.name} type="button" aria-pressed={on} onClick={() => toggleCuisine(c.name)} className={`chip ${on ? 'chip-on' : ''}`}>
                        {c.name}
                      </button>
                    );
                  })}
                </div>
                {meta.status === 'fallback' && (
                  <p className="mt-3 text-sm leading-relaxed text-slate-500">
                    Could not load the full lists from the server. Showing a short default list.
                  </p>
                )}
              </Filter>
              <Filter label="Course" value={selectedCourse}>
                <div role="radiogroup" aria-label="Course" className="flex flex-wrap gap-2">
                  {meta.courses.map((c) => {
                    const on = selectedCourse === c.name;
                    return (
                      <button
                        key={c.name}
                        type="button"
                        role="radio"
                        aria-checked={on}
                        onClick={() => setSelectedCourse(on ? '' : c.name)}
                        className={`chip ${on ? 'chip-on' : ''}`}
                      >
                        {c.name}
                      </button>
                    );
                  })}
                </div>
              </Filter>
              <Filter label="Model" value={modelLabel}>
                {(close) => (
                  <>
                    <ModelPicker models={meta.models} value={model} onChange={(id) => { changeModel(id); close(); }} />
                    <p className="mt-3 text-sm leading-relaxed text-slate-500">
                      Which model ranks the results. Switch after a search to compare their timings.
                    </p>
                  </>
                )}
              </Filter>
            </>
          )}
          <button type="button" disabled={busy} className="btn-primary w-full sm:ml-auto sm:w-auto" onClick={() => run()}>
            {busy ? 'Finding recipes...' : 'Find recipes'}
          </button>
        </div>
      </div>

      {hint && <p className="mt-3 px-1 text-[15px] text-red-700" role="alert">{hint}</p>}
      {likedCount > 0 && mode === 'ingredients' && (
        <p className="mt-2 flex flex-wrap items-center gap-x-3 px-1 text-[15px] text-slate-600">
          Tuned to your likes ({likedCount} liked)
          <button type="button" onClick={onClearLikes} className="min-h-11 text-red-700 underline-offset-4 hover:underline lg:min-h-9">
            Clear likes
          </button>
        </p>
      )}
    </div>
  );
};

export default SearchPanel;
