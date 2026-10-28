import type { ClassifierModel, OpenAICompletionsCompat, ThinkingLevelMap } from "../../src/types.ts";
import { defineProvider, modelsDevModels } from "../define.ts";
import { getEffortThinkingLevelMap } from "../reasoning.ts";

const BASE_COMPAT: OpenAICompletionsCompat = {
	supportsStore: false,
	supportsDeveloperRole: false,
	supportsReasoningEffort: false,
	supportsUsageInStreaming: true,
	maxTokensField: "max_tokens",
	supportsStrictMode: true,
	// Baseten automatic prompt caching needs session affinity so related
	// requests land on the same replica. See:
	// https://docs.baseten.co/inference/model-apis/pricing-and-limits
	sendSessionAffinityHeaders: true,
	supportsLongCacheRetention: false,
};
const REASONING_EFFORT_COMPAT: OpenAICompletionsCompat = {
	...BASE_COMPAT,
	supportsReasoningEffort: true,
	thinkingFormat: "openai",
};
const TOGGLE_REASONING_COMPAT: OpenAICompletionsCompat = {
	...BASE_COMPAT,
	thinkingFormat: "baseten",
	chatTemplateArgs: { enable_thinking: { $var: "thinking.enabled" } },
};
const TOGGLE_REASONING_EFFORT_COMPAT: OpenAICompletionsCompat = {
	...REASONING_EFFORT_COMPAT,
	thinkingFormat: "baseten",
	chatTemplateArgs: { enable_thinking: { $var: "thinking.enabled" } },
};
const TOGGLE_LEVELS: ThinkingLevelMap = {
	off: "off",
	minimal: null,
	low: null,
	medium: null,
	high: "high",
	xhigh: null,
	max: null,
};
const GLM_52_LEVELS: ThinkingLevelMap = {
	off: "none",
	minimal: null,
	low: null,
	medium: null,
	high: "high",
	xhigh: null,
	max: "max",
};
const GLM_52_MODEL_IDS = new Set(["zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast"]);

const BASE_URL = "https://inference.baseten.co/v1";

function classifier(id: string, name: string, input: number): ClassifierModel<"typesafe-system-one"> {
	return {
		type: "classifier",
		id,
		name,
		api: "typesafe-system-one",
		provider: "baseten",
		baseUrl: BASE_URL,
		input: ["text"],
		cost: { input, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 16384,
	};
}

// Base Labs decision models on Qwen3.5, served with TypeSafe's System One protocol at
// /v1/systemone. models.dev's v1 decision list does not carry them, so they are listed here.
// IDs and prices from models.dev's v2 baseten entries. models.dev also lists image input, which
// is not verified on Baseten's endpoint, so they are text-only here.
const CLASSIFIER_MODELS = [
	classifier("decision-models/baseten-elephant-9b-latest", "Elephant 1", 0.035),
	classifier("decision-models/baseten-mammoth-27b-latest", "Mammoth 1", 0.1),
];

export default defineProvider({
	id: "baseten",
	models: (ctx) =>
		modelsDevModels(ctx, "baseten", {
			api: "openai-completions",
			baseUrl: BASE_URL,
			skipDeprecated: true,
			recordReasoning: false,
			map: (model, source) => {
				const options = source.reasoningOptions ?? [];
				const isGlm52 = GLM_52_MODEL_IDS.has(model.id);
				const supportsToggle = isGlm52 || options.some((option) => option.type === "toggle");
				const supportsEffort = isGlm52 || options.some((option) => option.type === "effort");
				const compat =
					supportsToggle && supportsEffort
						? TOGGLE_REASONING_EFFORT_COMPAT
						: supportsToggle
							? TOGGLE_REASONING_COMPAT
							: supportsEffort
								? REASONING_EFFORT_COMPAT
								: BASE_COMPAT;
				const thinkingLevelMap = isGlm52
					? GLM_52_LEVELS
					: supportsToggle
						? TOGGLE_LEVELS
						: getEffortThinkingLevelMap(options);
				return {
					...model,
					...(thinkingLevelMap ? { thinkingLevelMap } : {}),
					// Baseten's GLM-5.2 endpoints are text-only despite models.dev reporting image input.
					input: !isGlm52 && source.imageInput ? ["text", "image"] : ["text"],
					compat,
				};
			},
		}),
	classifiers: () => CLASSIFIER_MODELS,
	compat: { "openai-completions": { supportsStrictMode: true } },
});
