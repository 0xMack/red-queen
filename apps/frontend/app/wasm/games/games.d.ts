/* tslint:disable */
/* eslint-disable */

export class RandomPolicy {
    free(): void;
    [Symbol.dispose](): void;
    decide(observation: Float64Array): number;
    constructor(seed: number);
}

/**
 * One Snake game plus the observer its model reads.
 */
export class SnakeGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Render cells flattened as [x, y, label, x, y, label, ...] with label 0 body / 1 head / 2 food,
     * in render order (body behind the head first, then head, then food).
     */
    cells(): Int32Array;
    /**
     * `seed` is a u32 (JS numbers carry 53 bits; every seed the site generates fits in 32).
     */
    constructor(width: number, height: number, seed: number, observer_id: string);
    /**
     * What the model sees right now, under this game's observer.
     */
    observation(): Float64Array;
    reset(): void;
    /**
     * Relative action (-1 / 0 / +1). Returns the reward; `done` says whether the game ended.
     */
    step(action: number): number;
    readonly done: boolean;
    readonly height: number;
    readonly score: number;
    readonly width: number;
}

/**
 * `relative3.v1`: raw model outputs -> -1 / 0 / +1.
 */
export function decodeRelative3(outputs: Float64Array): number;

/**
 * The greedy baseline; reads snake/features.v1.
 */
export function greedyDecide(observation: Float64Array): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_randompolicy_free: (a: number, b: number) => void;
    readonly __wbg_snakegame_free: (a: number, b: number) => void;
    readonly decodeRelative3: (a: number, b: number) => number;
    readonly greedyDecide: (a: number, b: number) => number;
    readonly randompolicy_decide: (a: number, b: number, c: number) => number;
    readonly randompolicy_new: (a: number) => number;
    readonly snakegame_cells: (a: number) => [number, number];
    readonly snakegame_done: (a: number) => number;
    readonly snakegame_height: (a: number) => number;
    readonly snakegame_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly snakegame_observation: (a: number) => [number, number];
    readonly snakegame_reset: (a: number) => void;
    readonly snakegame_score: (a: number) => number;
    readonly snakegame_step: (a: number, b: number) => number;
    readonly snakegame_width: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
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
