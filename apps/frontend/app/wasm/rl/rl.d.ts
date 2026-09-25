/* tslint:disable */
/* eslint-disable */

/**
 * A Snake game for a demo to play the agent's greedy policy on, move by move, and draw.
 */
export class DemoGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Cells as `[x, y, label, ...]`, label 0 body / 1 head / 2 food, in render order (like the games module).
     */
    cells(): Int32Array;
    constructor(seed: number, observer_id: string);
    observation(): Float64Array;
    /**
     * Relative action index (0 left, 1 straight, 2 right); returns the reward.
     */
    step(action: number): number;
    readonly done: boolean;
    readonly score: number;
}

/**
 * What one call to `Trainer::train` did, as numbers JS can read without a serializer.
 */
export class Progress {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    entropy: number;
    /**
     * Episodes that ended during the call, and their mean return and game score (NaN if none ended).
     */
    episodes: number;
    /**
     * Current epsilon (NaN for an agent that doesn't explore that way) and rows ever updated (tabular; NaN otherwise).
     */
    epsilon: number;
    mean_return: number;
    mean_score: number;
    /**
     * A DQN's mean Q(s, a) and loss over this call's updates, and its updates so far (NaN otherwise).
     */
    q_mean: number;
    states_visited: number;
    td_loss: number;
    total_episodes: number;
    total_steps: number;
    updates: number;
}

/**
 * One agent learning in one environment (`snake/<observer>+relative3.v1` on 10x10, or `reach1d`) -- the engine of
 * Learn's live demos, the same `Trainer` the training jobs drive through PyO3.
 */
export class Trainer {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The agent's value for each action on `observation` (a table's row, a Q-network's outputs; empty if it has none).
     */
    actionValues(observation: Float64Array): Float64Array;
    /**
     * The greedy policy's game score on each of `seeds`, games capped at `max_steps`.
     */
    evaluate(seeds: Uint32Array, max_steps: number): Float64Array;
    /**
     * The greedy action on `observation`, as an index (Snake: 0 left, 1 straight, 2 right).
     */
    greedyAction(observation: Float64Array): number;
    /**
     * `params`: the algorithm's hyperparameters as `name=value` pairs separated by commas (`"alpha=0.1,n_step=3"`,
     * empty for the defaults). `reward`: `shaped` or `sparse` (Snake).
     */
    constructor(algorithm: string, env_id: string, seed: number, params: string, reward: string);
    /**
     * A tabular agent's values, `row * actions + action` (empty for other agents).
     */
    qValues(): Float64Array;
    /**
     * The table row `observation` falls in (-1 for a non-tabular agent).
     */
    row(observation: Float64Array): number;
    /**
     * The current policy as its champion JSON (`modelpack.champions`).
     */
    snapshot(): string;
    /**
     * Advance by `steps` environment steps.
     */
    train(steps: number): Progress;
    /**
     * How many updates each row of the table has had (empty for other agents).
     */
    visits(): Uint32Array;
    /**
     * Environment steps trained so far.
     */
    readonly totalSteps: number;
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
 * `dqn_digest(seed)`: a DQN with every stability piece trained on Snake's egocentric observer, hashed.
 */
export function dqnDigest(seed: number): string;

/**
 * `learning_digest(seed)`: Q-learning and SARSA trained on Snake, hashed.
 */
export function learningDigest(seed: number): string;

/**
 * `pg_digest(seed)`: PPO (softmax and Gaussian) and REINFORCE with a baseline, trained and hashed.
 */
export function pgDigest(seed: number): string;

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
    readonly __wbg_demogame_free: (a: number, b: number) => void;
    readonly __wbg_get_progress_entropy: (a: number) => number;
    readonly __wbg_get_progress_episodes: (a: number) => number;
    readonly __wbg_get_progress_epsilon: (a: number) => number;
    readonly __wbg_get_progress_mean_return: (a: number) => number;
    readonly __wbg_get_progress_mean_score: (a: number) => number;
    readonly __wbg_get_progress_q_mean: (a: number) => number;
    readonly __wbg_get_progress_states_visited: (a: number) => number;
    readonly __wbg_get_progress_td_loss: (a: number) => number;
    readonly __wbg_get_progress_total_episodes: (a: number) => number;
    readonly __wbg_get_progress_total_steps: (a: number) => number;
    readonly __wbg_get_progress_updates: (a: number) => number;
    readonly __wbg_progress_free: (a: number, b: number) => void;
    readonly __wbg_set_progress_entropy: (a: number, b: number) => void;
    readonly __wbg_set_progress_episodes: (a: number, b: number) => void;
    readonly __wbg_set_progress_epsilon: (a: number, b: number) => void;
    readonly __wbg_set_progress_mean_return: (a: number, b: number) => void;
    readonly __wbg_set_progress_mean_score: (a: number, b: number) => void;
    readonly __wbg_set_progress_q_mean: (a: number, b: number) => void;
    readonly __wbg_set_progress_states_visited: (a: number, b: number) => void;
    readonly __wbg_set_progress_td_loss: (a: number, b: number) => void;
    readonly __wbg_set_progress_total_episodes: (a: number, b: number) => void;
    readonly __wbg_set_progress_total_steps: (a: number, b: number) => void;
    readonly __wbg_set_progress_updates: (a: number, b: number) => void;
    readonly __wbg_trainer_free: (a: number, b: number) => void;
    readonly benchForwards: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly benchUpdates: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly demogame_cells: (a: number) => [number, number];
    readonly demogame_done: (a: number) => number;
    readonly demogame_new: (a: number, b: number, c: number) => [number, number, number];
    readonly demogame_observation: (a: number) => [number, number];
    readonly demogame_score: (a: number) => number;
    readonly demogame_step: (a: number, b: number) => number;
    readonly dqnDigest: (a: number) => [number, number];
    readonly learningDigest: (a: number) => [number, number];
    readonly pgDigest: (a: number) => [number, number];
    readonly rolloutDigest: (a: number) => [number, number];
    readonly trainer_actionValues: (a: number, b: number, c: number) => [number, number];
    readonly trainer_evaluate: (a: number, b: number, c: number, d: number) => [number, number];
    readonly trainer_greedyAction: (a: number, b: number, c: number) => number;
    readonly trainer_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly trainer_qValues: (a: number) => [number, number];
    readonly trainer_row: (a: number, b: number, c: number) => number;
    readonly trainer_snapshot: (a: number) => [number, number];
    readonly trainer_totalSteps: (a: number) => number;
    readonly trainer_train: (a: number, b: number) => number;
    readonly trainer_visits: (a: number) => [number, number];
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
