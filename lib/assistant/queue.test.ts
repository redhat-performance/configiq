import { describe, expect, it } from 'vitest';
import { AssistantBusyError, createAssistantQueue } from './queue';

function deferred<T = void>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('assistant queue', () => {
  it('runs two tasks at once and queues the third', async () => {
    const queue = createAssistantQueue({ maxConcurrent: 2, maxWaitMs: 1000 });
    const first = deferred();
    const second = deferred();

    const a = queue.run(() => first.promise);
    const b = queue.run(() => second.promise);
    expect(queue.active()).toBe(2);

    let thirdStarted = false;
    const c = queue.run(async () => {
      thirdStarted = true;
    });

    await Promise.resolve();
    expect(thirdStarted).toBe(false);

    first.resolve();
    await a;
    await c;
    expect(thirdStarted).toBe(true);

    second.resolve();
    await b;
    expect(queue.active()).toBe(0);
  });

  it('refuses a request that waits too long', async () => {
    const queue = createAssistantQueue({ maxConcurrent: 1, maxWaitMs: 5 });
    const held = deferred();

    const first = queue.run(() => held.promise);
    await Promise.resolve();

    await expect(queue.run(async () => undefined)).rejects.toBeInstanceOf(AssistantBusyError);
    held.resolve();
    await first;
  });

  it('gives the slot back when a task throws', async () => {
    const queue = createAssistantQueue({ maxConcurrent: 1, maxWaitMs: 100 });
    await expect(queue.run(async () => { throw new Error('boom'); })).rejects.toThrow('boom');
    expect(queue.active()).toBe(0);
    await expect(queue.run(async () => 'ok')).resolves.toBe('ok');
  });

  it('releases a stream slot idempotently', async () => {
    const queue = createAssistantQueue({ maxConcurrent: 1, maxWaitMs: 100 });
    const release = await queue.acquire();
    expect(queue.active()).toBe(1);
    release();
    release();
    expect(queue.active()).toBe(0);
  });
});
