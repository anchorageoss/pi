import { openAICompletionsApi } from "../api/openai-completions.lazy.ts";
import { typesafeSystemOneApi } from "../api/typesafe-system-one.lazy.ts";
import { envApiKeyAuth } from "../auth/helpers.ts";
import { createProvider, type Provider } from "../models.ts";
import { BASETEN_CLASSIFIER_MODELS, BASETEN_MODELS } from "./baseten.models.ts";

export function basetenProvider(): Provider<"openai-completions"> {
	return createProvider<"openai-completions">({
		id: "baseten",
		name: "Baseten",
		baseUrl: "https://inference.baseten.co/v1",
		auth: { apiKey: envApiKeyAuth("Baseten API key", ["BASETEN_API_KEY"]) },
		models: [...Object.values(BASETEN_MODELS), ...Object.values(BASETEN_CLASSIFIER_MODELS)],
		api: openAICompletionsApi(),
		// Baseten serves its decision models over TypeSafe's System One protocol at /v1/systemone.
		classifiers: { "typesafe-system-one": typesafeSystemOneApi() },
	});
}
