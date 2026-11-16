/**
 * Inference concurrency guard.
 *
 * The assistant shares one small model on one GPU with everything else on the
 * host. Two concurrent questions is the limit that keeps a request from waiting
 * behind a queue longer than it would take to answer a third. Beyond the wait,
 * the route answers `503` with a message that points at search, which still
 * works when the model does not.
 */

export class AssistantBusyError extends Error {
  constructor(message = 'The assistant is busy. Search still works.') {
    super(message);
    this.name = 'AssistantBusyError';
  }
}

export interface AssistantQueueOptions {
  maxConcurrent: number;
  /** How long a request waits for a slot before it is refused. */
  maxWaitMs: number;
}

interface Waiter {
  resolve: () => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export interface AssistantQueue {
  /** Number of tasks currently holding a slot. */
  active(): number;
  /**
   * Take a slot and return the function that gives it back. Callers that
   * stream a response must release when the stream ends, not when the handler
   * returns.
   */
  acquire(): Promise<() => void>;
  /** Run `task` once a slot is free. Rejects with `AssistantBusyError`. */
  run<T>(task: () => Promise<T>): Promise<T>;
}

export function createAssistantQueue(options: AssistantQueueOptions): AssistantQueue {
  let active = 0;
  const waiters: Waiter[] = [];

  function release(): void {
    active -= 1;
    const next = waiters.shift();
    if (!next) return;
    clearTimeout(next.timer);
    active += 1;
    next.resolve();
  }

  function acquire(): Promise<void> {
    if (active < options.maxConcurrent) {
      active += 1;
      return Promise.resolve();
    }
    return new Promise<void>((resolve, reject) => {
      const waiter: Waiter = {
        resolve,
        reject,
        timer: setTimeout(() => {
          const index = waiters.indexOf(waiter);
          if (index >= 0) waiters.splice(index, 1);
          reject(new AssistantBusyError());
        }, options.maxWaitMs),
      };
      waiters.push(waiter);
    });
  }

  return {
    active: () => active,
    async acquire(): Promise<() => void> {
      await acquire();
      let released = false;
      return () => {
        if (released) return;
        released = true;
        release();
      };
    },
    async run<T>(task: () => Promise<T>): Promise<T> {
      await acquire();
      try {
        return await task();
      } finally {
        release();
      }
    },
  };
}

/** Process-wide queue for the assistant route. */
export const assistantQueue = createAssistantQueue({ maxConcurrent: 2, maxWaitMs: 15_000 });
