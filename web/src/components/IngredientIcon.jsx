/* eslint-disable react/prop-types */
import { formOf, toLch, fromLch } from '../cover/plate';

// A small drawing of an ingredient, in its own colour. The shape comes from the same
// rule the cover uses (formOf), so paneer is cubes and spinach is a leaf in both places.

const NEUTRAL = '#cdc6b8';
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const tones = new Map();
function tonesOf(hex) {
  let hit = tones.get(hex);
  if (!hit) {
    const { l, c, h } = toLch(hex);
    hit = {
      base: hex,
      dark: fromLch(clamp(l - 0.2, 0.2, 0.9), c, h),
      light: fromLch(clamp(l + 0.09, 0.3, 0.97), c * 0.8, h),
    };
    tones.set(hex, hit);
  }
  return hit;
}

// Each form as (base, dark, light) -> shapes in a 24 x 24 box.
const FORMS = {
  leaf: (b, d) => (
    <>
      <path d="M4 20C4 10 10 4 20 4c0 10-6 16-16 16Z" fill={b} stroke={d} />
      <path d="M5.5 18.5L15 9" stroke={d} strokeWidth="1.1" fill="none" />
    </>
  ),
  cube: (b, d, l) => (
    <>
      <rect x="3.5" y="9" width="11" height="11" rx="1.6" transform="rotate(-9 9 14.5)" fill={b} stroke={d} />
      <rect x="13" y="3.5" width="7.5" height="7.5" rx="1.3" transform="rotate(12 16.75 7.25)" fill={l} stroke={d} />
    </>
  ),
  half: (b, d, l) => (
    <>
      <path d="M2.5 9.5a9.5 9.5 0 0 0 19 0Z" fill={b} stroke={d} />
      <path d="M6 9.5a6 6 0 0 0 12 0Z" fill={l} />
    </>
  ),
  round: (b, d, l) => (
    <>
      <circle cx="9.5" cy="14" r="7" fill={b} stroke={d} />
      <circle cx="9.5" cy="14" r="3.8" fill={l} />
      <circle cx="18" cy="7" r="4" fill={b} stroke={d} />
      <circle cx="18" cy="7" r="1.8" fill={l} />
    </>
  ),
  berry: (b, d, l) => (
    <>
      <circle cx="8" cy="9" r="4.6" fill={b} stroke={d} />
      <circle cx="16.5" cy="10.5" r="4" fill={b} stroke={d} />
      <circle cx="11.5" cy="17" r="4.4" fill={b} stroke={d} />
      <circle cx="6.8" cy="7.8" r="1.2" fill={l} />
      <circle cx="15.5" cy="9.4" r="1.1" fill={l} />
      <circle cx="10.3" cy="15.8" r="1.2" fill={l} />
    </>
  ),
  grain: (b, d) => (
    <>
      <ellipse cx="7" cy="7.5" rx="3.6" ry="1.9" transform="rotate(-28 7 7.5)" fill={b} stroke={d} />
      <ellipse cx="16.5" cy="6.5" rx="3.6" ry="1.9" transform="rotate(24 16.5 6.5)" fill={b} stroke={d} />
      <ellipse cx="12" cy="12.5" rx="3.6" ry="1.9" transform="rotate(-6 12 12.5)" fill={b} stroke={d} />
      <ellipse cx="6.5" cy="17" rx="3.6" ry="1.9" transform="rotate(30 6.5 17)" fill={b} stroke={d} />
      <ellipse cx="17" cy="17.5" rx="3.6" ry="1.9" transform="rotate(-34 17 17.5)" fill={b} stroke={d} />
    </>
  ),
  strand: (b, d) => (
    <>
      {['M4 6c4-3 6 3 10 0s4-2 6-1', 'M4 12c4-3 6 3 10 0s4-2 6-1', 'M4 18c4-3 6 3 10 0s4-2 6-1'].map((path) => (
        <g key={path} fill="none" strokeLinecap="round">
          <path d={path} stroke={d} strokeWidth="3.3" />
          <path d={path} stroke={b} strokeWidth="2.1" />
        </g>
      ))}
    </>
  ),
  chilli: (b, d) => (
    <>
      <path d="M4.5 7C5 15.5 11 20.5 20.5 19.5C14.5 18 9.5 13.5 8.5 6.5Z" fill={b} stroke={d} strokeLinejoin="round" />
      <path d="M6.5 6.8L5.2 3.2" stroke={d} strokeWidth="1.6" strokeLinecap="round" fill="none" />
    </>
  ),
  powder: (b, d) => (
    <>
      <path d="M2.5 19.5c2.5-1 4.5-8.5 9.5-8.5s7 7.5 9.5 8.5Z" fill={b} stroke={d} strokeLinejoin="round" />
      <circle cx="8" cy="7" r="1.1" fill={d} />
      <circle cx="12.5" cy="4.5" r="1.1" fill={d} />
      <circle cx="16.5" cy="7.5" r="1.1" fill={d} />
    </>
  ),
  liquid: (b, d, l) => (
    <>
      <path d="M12 2.8c4 5 6.6 8.2 6.6 11.8a6.6 6.6 0 0 1-13.2 0c0-3.6 2.6-6.8 6.6-11.8Z" fill={b} stroke={d} strokeLinejoin="round" />
      <path d="M8.6 14.6a3.4 3.4 0 0 0 2.6 3.3" stroke={l} strokeWidth="1.5" strokeLinecap="round" fill="none" />
    </>
  ),
};

const IngredientIcon = ({ name, color, className = '' }) => {
  const { base, dark, light } = tonesOf(color || NEUTRAL);
  const draw = FORMS[formOf(name)] || FORMS.grain;
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" strokeWidth="0.9" className={`h-5 w-5 shrink-0 ${className}`}>
      {draw(base, dark, light)}
    </svg>
  );
};

export default IngredientIcon;
