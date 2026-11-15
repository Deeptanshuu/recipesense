/* eslint-disable react/prop-types */
import { useEffect, useState } from 'react';
import StepIcon from './StepIcon';
import { STEP_MS } from '../pacing';

const SLOW_MS = 5000;

const TEXT = { done: 'text-slate-500', active: 'text-ink', todo: 'text-slate-400' };
const ICON = { done: 'text-slate-400', active: 'step-on text-ink', todo: 'text-slate-300' };

// A loading state that says what the server is doing: the steps of the request, each with
// its own small drawing, lit one after another. Only the running step's drawing moves.
// A step is { icon, text }, or just the text. The time waited so far sits beside the
// running step. The steps move on a clock,
// not on news from the server, so the last one stays lit until the answer arrives.
const Pipeline = ({ steps, label, className = '' }) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const timer = setInterval(() => setElapsed(performance.now() - start), 100);
    return () => clearInterval(timer);
  }, []);

  const active = Math.min(steps.length - 1, Math.floor(elapsed / (STEP_MS || 650)));

  return (
    <div className={className}>
      <span className="sr-only" role="status">{label}</span>
      <ol aria-hidden="true" className="space-y-2">
        {steps.map((item, i) => {
          const { icon, text } = typeof item === 'string' ? { icon: 'embed', text: item } : item;
          const state = i < active ? 'done' : i === active ? 'active' : 'todo';
          return (
            <li key={text} className={`flex items-start gap-3 text-[15px] leading-6 transition-colors duration-300 ${TEXT[state]}`}>
              <StepIcon kind={icon} className={`transition-colors duration-300 ${ICON[state]}`} />
              <span className="min-w-0">
                {text}
                {state === 'active' && (
                  <span className="ml-2 whitespace-nowrap font-medium tabular-nums text-accent">
                    {(elapsed / 1000).toFixed(1)} s
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
      {elapsed > SLOW_MS && (
        <p className="mt-4 max-w-md text-[15px] leading-relaxed text-slate-600">
          Still working. If the server was asleep it loads its models first, which takes about ten
          seconds. After that a search takes milliseconds.
        </p>
      )}
    </div>
  );
};

export default Pipeline;
