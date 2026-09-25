/* @ts-self-types="./rl.d.ts" */

/**
 * A Snake game for a demo to play the agent's greedy policy on, move by move, and draw.
 */
export class DemoGame {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DemoGameFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_demogame_free(ptr, 0);
    }
    /**
     * Cells as `[x, y, label, ...]`, label 0 body / 1 head / 2 food, in render order (like the games module).
     * @returns {Int32Array}
     */
    cells() {
        const ret = wasm.demogame_cells(this.__wbg_ptr);
        var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {boolean}
     */
    get done() {
        const ret = wasm.demogame_done(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {number} seed
     * @param {string} observer_id
     */
    constructor(seed, observer_id) {
        const ptr0 = passStringToWasm0(observer_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.demogame_new(seed, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        DemoGameFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Float64Array}
     */
    observation() {
        const ret = wasm.demogame_observation(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get score() {
        const ret = wasm.demogame_score(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Relative action index (0 left, 1 straight, 2 right); returns the reward.
     * @param {number} action
     * @returns {number}
     */
    step(action) {
        const ret = wasm.demogame_step(this.__wbg_ptr, action);
        return ret;
    }
}
if (Symbol.dispose) DemoGame.prototype[Symbol.dispose] = DemoGame.prototype.free;

/**
 * What one call to `Trainer::train` did, as numbers JS can read without a serializer.
 */
export class Progress {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Progress.prototype);
        obj.__wbg_ptr = ptr;
        ProgressFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ProgressFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_progress_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get entropy() {
        const ret = wasm.__wbg_get_progress_entropy(this.__wbg_ptr);
        return ret;
    }
    /**
     * Episodes that ended during the call, and their mean return and game score (NaN if none ended).
     * @returns {number}
     */
    get episodes() {
        const ret = wasm.__wbg_get_progress_episodes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Current epsilon (NaN for an agent that doesn't explore that way) and rows ever updated (tabular; NaN otherwise).
     * @returns {number}
     */
    get epsilon() {
        const ret = wasm.__wbg_get_progress_epsilon(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get mean_return() {
        const ret = wasm.__wbg_get_progress_mean_return(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get mean_score() {
        const ret = wasm.__wbg_get_progress_mean_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * A DQN's mean Q(s, a) and loss over this call's updates, and its updates so far (NaN otherwise).
     * @returns {number}
     */
    get q_mean() {
        const ret = wasm.__wbg_get_progress_q_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get states_visited() {
        const ret = wasm.__wbg_get_progress_states_visited(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get td_loss() {
        const ret = wasm.__wbg_get_progress_td_loss(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get total_episodes() {
        const ret = wasm.__wbg_get_progress_total_episodes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get total_steps() {
        const ret = wasm.__wbg_get_progress_total_steps(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get updates() {
        const ret = wasm.__wbg_get_progress_updates(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set entropy(arg0) {
        wasm.__wbg_set_progress_entropy(this.__wbg_ptr, arg0);
    }
    /**
     * Episodes that ended during the call, and their mean return and game score (NaN if none ended).
     * @param {number} arg0
     */
    set episodes(arg0) {
        wasm.__wbg_set_progress_episodes(this.__wbg_ptr, arg0);
    }
    /**
     * Current epsilon (NaN for an agent that doesn't explore that way) and rows ever updated (tabular; NaN otherwise).
     * @param {number} arg0
     */
    set epsilon(arg0) {
        wasm.__wbg_set_progress_epsilon(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set mean_return(arg0) {
        wasm.__wbg_set_progress_mean_return(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set mean_score(arg0) {
        wasm.__wbg_set_progress_mean_score(this.__wbg_ptr, arg0);
    }
    /**
     * A DQN's mean Q(s, a) and loss over this call's updates, and its updates so far (NaN otherwise).
     * @param {number} arg0
     */
    set q_mean(arg0) {
        wasm.__wbg_set_progress_q_mean(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set states_visited(arg0) {
        wasm.__wbg_set_progress_states_visited(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set td_loss(arg0) {
        wasm.__wbg_set_progress_td_loss(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set total_episodes(arg0) {
        wasm.__wbg_set_progress_total_episodes(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set total_steps(arg0) {
        wasm.__wbg_set_progress_total_steps(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set updates(arg0) {
        wasm.__wbg_set_progress_updates(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) Progress.prototype[Symbol.dispose] = Progress.prototype.free;

/**
 * One agent learning in one environment (`snake/<observer>+relative3.v1` on 10x10, or `reach1d`) -- the engine of
 * Learn's live demos, the same `Trainer` the training jobs drive through PyO3.
 */
export class Trainer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TrainerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trainer_free(ptr, 0);
    }
    /**
     * The agent's value for each action on `observation` (a table's row, a Q-network's outputs; empty if it has none).
     * @param {Float64Array} observation
     * @returns {Float64Array}
     */
    actionValues(observation) {
        const ptr0 = passArrayF64ToWasm0(observation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trainer_actionValues(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * The greedy policy's game score on each of `seeds`, games capped at `max_steps`.
     * @param {Uint32Array} seeds
     * @param {number} max_steps
     * @returns {Float64Array}
     */
    evaluate(seeds, max_steps) {
        const ptr0 = passArray32ToWasm0(seeds, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trainer_evaluate(this.__wbg_ptr, ptr0, len0, max_steps);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * The greedy action on `observation`, as an index (Snake: 0 left, 1 straight, 2 right).
     * @param {Float64Array} observation
     * @returns {number}
     */
    greedyAction(observation) {
        const ptr0 = passArrayF64ToWasm0(observation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trainer_greedyAction(this.__wbg_ptr, ptr0, len0);
        return ret >>> 0;
    }
    /**
     * `params`: the algorithm's hyperparameters as `name=value` pairs separated by commas (`"alpha=0.1,n_step=3"`,
     * empty for the defaults). `reward`: `shaped` or `sparse` (Snake).
     * @param {string} algorithm
     * @param {string} env_id
     * @param {number} seed
     * @param {string} params
     * @param {string} reward
     */
    constructor(algorithm, env_id, seed, params, reward) {
        const ptr0 = passStringToWasm0(algorithm, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(env_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(params, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(reward, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.trainer_new(ptr0, len0, ptr1, len1, seed, ptr2, len2, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        TrainerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * A tabular agent's values, `row * actions + action` (empty for other agents).
     * @returns {Float64Array}
     */
    qValues() {
        const ret = wasm.trainer_qValues(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * The table row `observation` falls in (-1 for a non-tabular agent).
     * @param {Float64Array} observation
     * @returns {number}
     */
    row(observation) {
        const ptr0 = passArrayF64ToWasm0(observation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.trainer_row(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * The current policy as its champion JSON (`modelpack.champions`).
     * @returns {string}
     */
    snapshot() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.trainer_snapshot(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Environment steps trained so far.
     * @returns {number}
     */
    get totalSteps() {
        const ret = wasm.trainer_totalSteps(this.__wbg_ptr);
        return ret;
    }
    /**
     * Advance by `steps` environment steps.
     * @param {number} steps
     * @returns {Progress}
     */
    train(steps) {
        const ret = wasm.trainer_train(this.__wbg_ptr, steps);
        return Progress.__wrap(ret);
    }
    /**
     * How many updates each row of the table has had (empty for other agents).
     * @returns {Uint32Array}
     */
    visits() {
        const ret = wasm.trainer_visits(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) Trainer.prototype[Symbol.dispose] = Trainer.prototype.free;

/**
 * `iterations` single-observation forward passes (choosing an action). Returns a checksum; time the call.
 * @param {Uint32Array} layer_sizes
 * @param {string} activations
 * @param {number} iterations
 * @returns {number}
 */
export function benchForwards(layer_sizes, activations, iterations) {
    const ptr0 = passArray32ToWasm0(layer_sizes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(activations, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.benchForwards(ptr0, len0, ptr1, len1, iterations);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * `iterations` training updates (forward + backward + Adam) on a batch. `activations`: comma-separated, one per
 * layer after the input (`"relu,relu,linear"`). Returns a checksum; time the call.
 * @param {Uint32Array} layer_sizes
 * @param {string} activations
 * @param {number} batch
 * @param {number} iterations
 * @returns {number}
 */
export function benchUpdates(layer_sizes, activations, batch, iterations) {
    const ptr0 = passArray32ToWasm0(layer_sizes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(activations, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.benchUpdates(ptr0, len0, ptr1, len1, batch, iterations);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * `dqn_digest(seed)`: a DQN with every stability piece trained on Snake's egocentric observer, hashed.
 * @param {number} seed
 * @returns {string}
 */
export function dqnDigest(seed) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.dqnDigest(seed);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * `learning_digest(seed)`: Q-learning and SARSA trained on Snake, hashed.
 * @param {number} seed
 * @returns {string}
 */
export function learningDigest(seed) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.learningDigest(seed);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * `rollout_digest(seed)`: a random agent trained and evaluated on Snake and Reach1D, hashed.
 * @param {number} seed
 * @returns {string}
 */
export function rolloutDigest(seed) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.rolloutDigest(seed);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * `training_digest(seed, updates)` of `redqueen_rl::digest`: must equal the native build's (determinism.json).
 * @param {number} seed
 * @param {number} updates
 * @returns {string}
 */
export function trainingDigest(seed, updates) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.trainingDigest(seed, updates);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_8c4e43fe74559d73: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./rl_bg.js": import0,
    };
}

const DemoGameFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_demogame_free(ptr >>> 0, 1));
const ProgressFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_progress_free(ptr >>> 0, 1));
const TrainerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trainer_free(ptr >>> 0, 1));

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat64ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }


    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
