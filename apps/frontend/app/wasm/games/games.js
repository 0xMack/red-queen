/* @ts-self-types="./games.d.ts" */

/**
 * Checkers for the browser. Moves cross as indexes into the current `legalMoves()` list -- the same
 * list, in the same order, the Python side and every strategy see.
 */
export class CheckersGame {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CheckersGameFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_checkersgame_free(ptr, 0);
    }
    /**
     * Flattened [x, y, piece] with piece 0 red man / 1 red king / 2 black man / 3 black king.
     * @returns {Int32Array}
     */
    cells() {
        const ret = wasm.checkersgame_cells(this.__wbg_ptr);
        var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get currentPlayer() {
        const ret = wasm.checkersgame_currentPlayer(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get done() {
        const ret = wasm.checkersgame_done(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Flattened: for each move, its square count n, then n (x, y) pairs.
     * @returns {Int32Array}
     */
    legalMoves() {
        const ret = wasm.checkersgame_legalMoves(this.__wbg_ptr);
        var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {number} max_moves_without_capture
     */
    constructor(max_moves_without_capture) {
        const ret = wasm.checkersgame_new(max_moves_without_capture);
        this.__wbg_ptr = ret >>> 0;
        CheckersGameFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Float64Array}
     */
    observation() {
        const ret = wasm.checkersgame_observation(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    reset() {
        wasm.checkersgame_reset(this.__wbg_ptr);
    }
    /**
     * @param {number} index
     * @returns {Float64Array}
     */
    simulate(index) {
        const ret = wasm.checkersgame_simulate(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Play the `index`-th legal move; returns whether the game is over.
     * @param {number} index
     * @returns {boolean}
     */
    step(index) {
        const ret = wasm.checkersgame_step(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * -1 while undecided or for a draw; check `done` to tell them apart.
     * @returns {number}
     */
    get winner() {
        const ret = wasm.checkersgame_winner(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) CheckersGame.prototype[Symbol.dispose] = CheckersGame.prototype.free;

export class RandomPolicy {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RandomPolicyFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_randompolicy_free(ptr, 0);
    }
    /**
     * @param {Float64Array} observation
     * @returns {number}
     */
    decide(observation) {
        const ptr0 = passArrayF64ToWasm0(observation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.randompolicy_decide(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {number} seed
     */
    constructor(seed) {
        const ret = wasm.randompolicy_new(seed);
        this.__wbg_ptr = ret >>> 0;
        RandomPolicyFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) RandomPolicy.prototype[Symbol.dispose] = RandomPolicy.prototype.free;

export class Reach1DGame {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Reach1DGameFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_reach1dgame_free(ptr, 0);
    }
    /**
     * @param {number} target
     * @param {number} start_position
     * @param {number} start_velocity
     * @param {number} dt
     * @param {number} max_acceleration
     * @param {number} damping
     */
    constructor(target, start_position, start_velocity, dt, max_acceleration, damping) {
        const ret = wasm.reach1dgame_new(target, start_position, start_velocity, dt, max_acceleration, damping);
        this.__wbg_ptr = ret >>> 0;
        Reach1DGameFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * [position - target, velocity]
     * @returns {Float64Array}
     */
    observation() {
        const ret = wasm.reach1dgame_observation(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get position() {
        const ret = wasm.reach1dgame_position(this.__wbg_ptr);
        return ret;
    }
    reset() {
        wasm.reach1dgame_reset(this.__wbg_ptr);
    }
    /**
     * Returns the reward.
     * @param {number} action
     * @returns {number}
     */
    step(action) {
        const ret = wasm.reach1dgame_step(this.__wbg_ptr, action);
        return ret;
    }
}
if (Symbol.dispose) Reach1DGame.prototype[Symbol.dispose] = Reach1DGame.prototype.free;

/**
 * One Snake game plus the observer its model reads.
 */
export class SnakeGame {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SnakeGameFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_snakegame_free(ptr, 0);
    }
    /**
     * Render cells flattened as [x, y, label, x, y, label, ...] with label 0 body / 1 head / 2 food,
     * in render order (body behind the head first, then head, then food).
     * @returns {Int32Array}
     */
    cells() {
        const ret = wasm.snakegame_cells(this.__wbg_ptr);
        var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {boolean}
     */
    get done() {
        const ret = wasm.snakegame_done(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get height() {
        const ret = wasm.snakegame_height(this.__wbg_ptr);
        return ret;
    }
    /**
     * `seed` is a u32 (JS numbers carry 53 bits; every seed the site generates fits in 32).
     * @param {number} width
     * @param {number} height
     * @param {number} seed
     * @param {string} observer_id
     */
    constructor(width, height, seed, observer_id) {
        const ptr0 = passStringToWasm0(observer_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.snakegame_new(width, height, seed, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        SnakeGameFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * What the model sees right now, under this game's observer.
     * @returns {Float64Array}
     */
    observation() {
        const ret = wasm.snakegame_observation(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    reset() {
        wasm.snakegame_reset(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    get score() {
        const ret = wasm.snakegame_score(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Relative action (-1 / 0 / +1). Returns the reward; `done` says whether the game ended.
     * @param {number} action
     * @returns {number}
     */
    step(action) {
        const ret = wasm.snakegame_step(this.__wbg_ptr, action);
        return ret;
    }
    /**
     * @returns {number}
     */
    get width() {
        const ret = wasm.snakegame_width(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) SnakeGame.prototype[Symbol.dispose] = SnakeGame.prototype.free;

/**
 * `relative3.v1`: raw model outputs -> -1 / 0 / +1.
 * @param {Float64Array} outputs
 * @returns {number}
 */
export function decodeRelative3(outputs) {
    const ptr0 = passArrayF64ToWasm0(outputs, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.decodeRelative3(ptr0, len0);
    return ret;
}

/**
 * The greedy baseline; reads snake/features.v1.
 * @param {Float64Array} observation
 * @returns {number}
 */
export function greedyDecide(observation) {
    const ptr0 = passArrayF64ToWasm0(observation, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.greedyDecide(ptr0, len0);
    return ret;
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
        "./games_bg.js": import0,
    };
}

const CheckersGameFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_checkersgame_free(ptr >>> 0, 1));
const RandomPolicyFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_randompolicy_free(ptr >>> 0, 1));
const Reach1DGameFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_reach1dgame_free(ptr >>> 0, 1));
const SnakeGameFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_snakegame_free(ptr >>> 0, 1));

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
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

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
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
