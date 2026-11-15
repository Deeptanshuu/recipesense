import { useEffect, useState } from 'react';
import { checkHealth } from '../api';

const ATTEMPTS = 3;
const RETRY_DELAY = 2000;
const RECHECK_DELAY = 15000;

// Shown only when the health check keeps failing. A cold start of the API can
// take longer than one request, so one slow or failed check is not "offline".
const WarningHeader = () => {
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let timer = null;

    const wait = (ms) =>
      new Promise((resolve) => {
        timer = setTimeout(resolve, ms);
      });

    const probe = async () => {
      for (let attempt = 0; attempt < ATTEMPTS; attempt += 1) {
        try {
          await checkHealth({ signal: controller.signal });
          setOffline(false);
          return;
        } catch (err) {
          if (controller.signal.aborted) return;
        }
        if (attempt < ATTEMPTS - 1) await wait(RETRY_DELAY);
        if (controller.signal.aborted) return;
      }
      setOffline(true);
      // Keep checking, so the banner clears itself once the server answers.
      await wait(RECHECK_DELAY);
      if (!controller.signal.aborted) probe();
    };

    probe();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  if (!offline) return null;

  return (
    <div className="border-b border-red-100 bg-red-50 px-4 py-3 text-center text-[15px] text-red-800" role="alert">
      <strong className="font-semibold">Warning:</strong> Server is offline and not reachable. Please check your internet connection or try again later.
    </div>
  );
};

export default WarningHeader;
