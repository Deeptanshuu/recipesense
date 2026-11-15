/* eslint-disable react/prop-types */
import Pipeline from './Pipeline';

// A centred loading state: the steps of the request, or just the label when there are none.
const Loading = ({ label = 'Loading', steps }) => (
  <div className="flex justify-center px-5 py-16">
    <Pipeline label={label} steps={steps && steps.length ? steps : [label]} />
  </div>
);

export default Loading;
