/* tslint:disable */
/* eslint-disable */

/**
 * One agent learning in one environment (`snake/<observer>+relative3.v1` on 10x10, or `reach1d`).
 */
export class Trainer {
    free(): void;
    [Symbol.dispose](): void;
    constructor(algorithm: string, env_id: string, seed: number);
    /**
     * Advance by `steps` environment steps; returns how many episodes ended.
     */
    train(steps: number): number;
}

/**
 * `iterations` single-observation forward passes (choosing an action). Returns a checksum; time the call.
 */
export function benchForwards(layer_sizes: Uint32Array, activations: string, iterations: number): number;

/**
 * `iterations` training updates (forward + backward + Adam) on a batch. `activations`: comma-separated, one per
 * layer after the input (`"relu,relu,linear"`). Returns a checksum; time the call.
 */
export function benchUpdates(layer_sizes: Uint32Array, activations: string, batch: number, iterations: number): number;

/**
 * `rollout_digest(seed)`: a random agent trained and evaluated on Snake and Reach1D, hashed.
 */
export function rolloutDigest(seed: number): string;

/**
 * `training_digest(seed, updates)` of `redqueen_rl::digest`: must equal the native build's (determinism.json).
 */
export function trainingDigest(seed: number, updates: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_trainer_free: (a: number, b: number) => void;
    readonly benchForwards: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly benchUpdates: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly rolloutDigest: (a: number) => [number, number];
    readonly trainer_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly trainer_train: (a: number, b: number) => number;
    readonly trainingDigest: (a: number, b: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
