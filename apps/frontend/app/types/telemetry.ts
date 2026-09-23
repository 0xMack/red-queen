// apis/backend's run models -- libs/telemetry's pydantic RunInfo and GenerationStats, which the backend reuses
// directly as response models -- generated from the backend's OpenAPI schema (`api.gen.ts`, `pnpm gen:api-types`;
// apis/backend/tests/test_openapi_snapshot.py fails when the snapshot it's generated from is stale).

import type { components } from "./api.gen"

type Schemas = components["schemas"]

// A response always carries every field (pydantic serializes defaults too); the schema marks a field with a
// default as optional only because a *request* may omit it.
export type RunInfo = Required<Schemas["RunInfo"]>
export type RunStatus = RunInfo["status"]
export type GenerationStats = Required<Schemas["GenerationStats"]>
// One run as a runs list shows it (apis/backend's `GET /runs/summaries`): counts, best, last stats, a short trend.
export type RunSummary = Required<Schemas["RunSummary"]>
