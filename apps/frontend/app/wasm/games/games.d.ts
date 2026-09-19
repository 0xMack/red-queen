/* tslint:disable */
/* eslint-disable */

/**
 * Checkers for the browser. Moves cross as indexes into the current `legalMoves()` list -- the same
 * list, in the same order, the Python side and every strategy see.
 */
export class CheckersGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Flattened [x, y, piece] with piece 0 red man / 1 red king / 2 black man / 3 black king.
     */
    cells(): Int32Array;
    /**
     * Flattened: for each move, its square count n, then n (x, y) pairs.
     */
    legalMoves(): Int32Array;
    constructor(max_moves_without_capture: number);
    observation(): Float64Array;
    reset(): void;
    simulate(index: number): Float64Array;
    /**
     * Play the `index`-th legal move; returns whether the game is over.
     */
    step(index: number): boolean;
    readonly currentPlayer: number;
    readonly done: boolean;
    /**
     * -1 while undecided or for a draw; check `done` to tell them apart.
     */
    readonly winner: number;
}

export class RandomPolicy {
    free(): void;
    [Symbol.dispose](): void;
    decide(observation: Float64Array): number;
    constructor(seed: number);
}

export class Reach1DGame {
    free(): void;
    [Symbol.dispose](): void;
    constructor(target: number, start_position: number, start_velocity: number, dt: number, max_acceleration: number, damping: number);
    /**
     * [position - target, velocity]
     */
    observation(): Float64Array;
    reset(): void;
    /**
     * Returns the reward.
     */
    step(action: number): number;
    readonly position: number;
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
    readonly __wbg_checkersgame_free: (a: number, b: number) => void;
    readonly __wbg_randompolicy_free: (a: number, b: number) => void;
    readonly __wbg_reach1dgame_free: (a: number, b: number) => void;
    readonly __wbg_snakegame_free: (a: number, b: number) => void;
    readonly checkersgame_cells: (a: number) => [number, number];
    readonly checkersgame_currentPlayer: (a: number) => number;
    readonly checkersgame_done: (a: number) => number;
    readonly checkersgame_legalMoves: (a: number) => [number, number];
    readonly checkersgame_new: (a: number) => number;
    readonly checkersgame_observation: (a: number) => [number, number];
    readonly checkersgame_reset: (a: number) => void;
    readonly checkersgame_simulate: (a: number, b: number) => [number, number, number, number];
    readonly checkersgame_step: (a: number, b: number) => [number, number, number];
    readonly checkersgame_winner: (a: number) => number;
    readonly decodeRelative3: (a: number, b: number) => number;
    readonly greedyDecide: (a: number, b: number) => number;
    readonly randompolicy_decide: (a: number, b: number, c: number) => number;
    readonly randompolicy_new: (a: number) => number;
    readonly reach1dgame_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly reach1dgame_observation: (a: number) => [number, number];
    readonly reach1dgame_position: (a: number) => number;
    readonly reach1dgame_reset: (a: number) => void;
    readonly reach1dgame_step: (a: number, b: number) => number;
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
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
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
