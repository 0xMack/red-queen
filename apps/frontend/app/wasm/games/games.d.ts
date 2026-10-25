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
     * An independent copy of this game in its current position -- what a lookahead strategy steps
     * through to see a move's consequences without touching the real game.
     */
    duplicate(): CheckersGame;
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

/**
 * A Checkers strategy for the browser -- the same players training and evaluation use
 * (rust/core/src/checkers_strategies.rs). `pick` returns an index into the game's `legalMoves()`.
 */
export class CheckersStrategy {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Every layer's values, concatenated (input layer first; split by the network's layer sizes), when the
     * network evaluated the position legal move `index` leads to -- or undefined for a strategy without one.
     */
    activations(game: CheckersGame, index: number): Float64Array | undefined;
    /**
     * `name`: random | first-legal | material-N | evaluator. Only the evaluator takes a network: `weights` +
     * `layerSizes` (an evolve.WeightVector's) or `graph` (a compiled NEAT genome), searched `depth` plies
     * deep (1 = one ply; omit for the default). `quiescence` extends a search (material-N, evaluator) up to that
     * many plies through captures (docs/design/0021; omit or 0 for none).
     */
    constructor(name: string, seed: number, weights?: Float64Array | null, layer_sizes?: Uint32Array | null, depth?: number | null, graph?: Float64Array | null, quiescence?: number | null);
    pick(game: CheckersGame): number;
    /**
     * One score per legal move (higher = better), or undefined for a strategy that doesn't score moves.
     */
    scores(game: CheckersGame): Float64Array | undefined;
}

/**
 * Copy Cat (docs/design/0022), phase 1: an SVG read in the protocol's subset, rendered by the same rasterizer the
 * protocol scores with. The game page and its search moves come later.
 */
export class CopyCatGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The canonical serialization (0.1 px, a viewBox of the image's size): what shapes and bytes are counted from.
     */
    canonical(): string;
    /**
     * Shape `i`'s kind: "rect", "circle", "ellipse", "polygon" or "path".
     */
    kind(i: number): string;
    /**
     * The SVG of shape `i` alone, in the same frame (a layer, for an exploded view).
     */
    layer(i: number): string;
    /**
     * Throws, with the reason, for anything outside the subset.
     */
    constructor(svg: string);
    /**
     * Straight-alpha RGBA, 8 bits per channel: ready for an `ImageData`.
     */
    render(): Uint8Array;
    /**
     * The structured pipeline (docs/design/0022 step 2) on an opaque RGBA image (an `ImageData`'s bytes), with its
     * default settings. Throws if the bytes aren't `width x height` pixels.
     */
    static trace(rgba: Uint8Array, width: number, height: number): CopyCatGame;
    readonly height: number;
    readonly nodes: number;
    readonly palette: number;
    readonly shapes: number;
    readonly width: number;
}

/**
 * The structured pipeline with what each stage made (docs/design/0022 step 2), for drawing it at work.
 */
export class CopyCatTrace {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Per pixel, its label once the palette was merged, before cleaning (0 = background).
     */
    clustered(): Uint16Array;
    /**
     * The SVG it made.
     */
    document(): CopyCatGame;
    /**
     * Per pixel, its label after cleaning (0 = background).
     */
    labels(): Uint16Array;
    /**
     * On an opaque RGBA image (an `ImageData`'s bytes), with the default settings.
     */
    constructor(rgba: Uint8Array, width: number, height: number);
    /**
     * Per label, its fill as RGB bytes (label 0: the background's mean colour).
     */
    palette(): Uint8Array;
    /**
     * Whether the background was fitted as a gradient.
     */
    readonly gradient: boolean;
    /**
     * The noise's estimated standard deviation (OKLab).
     */
    readonly sigma: number;
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
    readonly __wbg_checkersstrategy_free: (a: number, b: number) => void;
    readonly __wbg_copycatgame_free: (a: number, b: number) => void;
    readonly __wbg_copycattrace_free: (a: number, b: number) => void;
    readonly __wbg_randompolicy_free: (a: number, b: number) => void;
    readonly __wbg_reach1dgame_free: (a: number, b: number) => void;
    readonly __wbg_snakegame_free: (a: number, b: number) => void;
    readonly checkersgame_cells: (a: number) => [number, number];
    readonly checkersgame_currentPlayer: (a: number) => number;
    readonly checkersgame_done: (a: number) => number;
    readonly checkersgame_duplicate: (a: number) => number;
    readonly checkersgame_legalMoves: (a: number) => [number, number];
    readonly checkersgame_new: (a: number) => number;
    readonly checkersgame_observation: (a: number) => [number, number];
    readonly checkersgame_reset: (a: number) => void;
    readonly checkersgame_simulate: (a: number, b: number) => [number, number, number, number];
    readonly checkersgame_step: (a: number, b: number) => [number, number, number];
    readonly checkersgame_winner: (a: number) => number;
    readonly checkersstrategy_activations: (a: number, b: number, c: number) => [number, number];
    readonly checkersstrategy_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number];
    readonly checkersstrategy_pick: (a: number, b: number) => [number, number, number];
    readonly checkersstrategy_scores: (a: number, b: number) => [number, number];
    readonly copycatgame_canonical: (a: number) => [number, number];
    readonly copycatgame_height: (a: number) => number;
    readonly copycatgame_kind: (a: number, b: number) => [number, number];
    readonly copycatgame_layer: (a: number, b: number) => [number, number, number, number];
    readonly copycatgame_new: (a: number, b: number) => [number, number, number];
    readonly copycatgame_nodes: (a: number) => number;
    readonly copycatgame_palette: (a: number) => number;
    readonly copycatgame_render: (a: number) => [number, number];
    readonly copycatgame_shapes: (a: number) => number;
    readonly copycatgame_trace: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly copycatgame_width: (a: number) => number;
    readonly copycattrace_clustered: (a: number) => [number, number];
    readonly copycattrace_document: (a: number) => number;
    readonly copycattrace_gradient: (a: number) => number;
    readonly copycattrace_labels: (a: number) => [number, number];
    readonly copycattrace_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly copycattrace_palette: (a: number) => [number, number];
    readonly copycattrace_sigma: (a: number) => number;
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
