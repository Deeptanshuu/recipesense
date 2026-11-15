/* eslint-disable react/prop-types */
// One small drawing per kind of pipeline step. Each is still by default and moves
// only inside an element with the class `step-on` (see the si-* rules in index.css),
// so the same drawing serves the waiting, running and finished states.

const delay = (s) => ({ animationDelay: `${s}s` });
const dot = (cx, cy, r, d, key) => (
  <circle key={key} cx={cx} cy={cy} r={r} fill="currentColor" stroke="none" className="si-pulse" style={delay(d)} />
);

const GRID = [6, 12, 18];

const ICONS = {
  // Lines of text written one after another.
  read: (
    <>
      <path d="M5 7h14" pathLength="1" className="si-draw" />
      <path d="M5 12h14" pathLength="1" className="si-draw" style={delay(0.2)} />
      <path d="M5 17h9" pathLength="1" className="si-draw" style={delay(0.4)} />
    </>
  ),
  // What was typed on the left, what the models know on the right, joined up.
  match: (
    <>
      <path d="M6 6L18 12" pathLength="1" className="si-draw" />
      <path d="M6 12L18 6" pathLength="1" className="si-draw" style={delay(0.2)} />
      <path d="M6 18H18" pathLength="1" className="si-draw" style={delay(0.4)} />
      {GRID.map((y) => <circle key={`l${y}`} cx="6" cy={y} r="1.7" fill="currentColor" stroke="none" />)}
      {GRID.map((y) => <circle key={`r${y}`} cx="18" cy={y} r="1.7" fill="currentColor" stroke="none" />)}
    </>
  ),
  // A vector: a grid of numbers lighting up in a wave.
  embed: <>{GRID.flatMap((y, j) => GRID.map((x, i) => dot(x, y, 1.8, (i + j) * 0.13, `${x}-${y}`)))}</>,
  // Three sources flowing into one list.
  retrieve: (
    <>
      <path d="M4 6C10 6 9 12 15 12" className="si-flow" />
      <path d="M4 12H15" className="si-flow" />
      <path d="M4 18C10 18 9 12 15 12" className="si-flow" />
      {dot(19, 12, 2.2, 0, 'out')}
    </>
  ),
  // A decision tree, lit from the root down.
  trees: (
    <>
      <path d="M12 5L6.5 12M12 5L17.5 12M6.5 12L4 19M6.5 12L9 19M17.5 12L15 19M17.5 12L20 19" strokeWidth="1.1" opacity="0.45" />
      {dot(12, 5, 2, 0, 'root')}
      {dot(6.5, 12, 1.8, 0.22, 'a')}
      {dot(17.5, 12, 1.8, 0.22, 'b')}
      {[4, 9, 15, 20].map((x) => dot(x, 19, 1.5, 0.44, x))}
    </>
  ),
  // A sweep over the catalogue, finding the points that are close.
  compare: (
    <>
      <circle cx="12" cy="12" r="8.5" strokeWidth="1.1" opacity="0.4" />
      <path d="M12 12V3.5" className="si-spin" />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
      {dot(8, 8.5, 1.2, 0.9, 'p1')}
      {dot(16.5, 14, 1.2, 0.3, 'p2')}
      {dot(8.5, 16, 1.2, 0.6, 'p3')}
    </>
  ),
  // Bars changing places.
  sort: (
    <>
      <rect x="4.6" y="12" width="3.4" height="7" rx="1" fill="currentColor" stroke="none" className="si-bar" />
      <rect x="10.3" y="6" width="3.4" height="13" rx="1" fill="currentColor" stroke="none" className="si-bar" style={delay(0.25)} />
      <rect x="16" y="9" width="3.4" height="10" rx="1" fill="currentColor" stroke="none" className="si-bar" style={delay(0.5)} />
    </>
  ),
  // Two sets and what they share.
  overlap: (
    <>
      <circle cx="9.5" cy="12" r="5.5" />
      <circle cx="14.5" cy="12" r="5.5" />
      <path d="M12 7.1a5.5 5.5 0 0 1 0 9.8a5.5 5.5 0 0 1 0-9.8Z" fill="currentColor" stroke="none" className="si-pulse" />
    </>
  ),
  // A balance: rare ingredients weigh more.
  weigh: (
    <>
      <path d="M12 6V19M8.5 19H15.5" />
      <g className="si-tilt">
        <path d="M5 7H19" />
        <circle cx="5" cy="11" r="2" />
        <circle cx="19" cy="11.6" r="2.6" />
      </g>
    </>
  ),
  // Dabs of colour.
  palette: (
    <>
      {dot(8, 9, 4, 0, 'c1')}
      {dot(16, 9, 4, 0.25, 'c2')}
      {dot(12, 16, 4, 0.5, 'c3')}
    </>
  ),
};

const StepIcon = ({ kind, className = '' }) => (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.7"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={`h-6 w-6 shrink-0 ${className}`}
  >
    {ICONS[kind] || ICONS.embed}
  </svg>
);

export default StepIcon;
