import { afterEach, describe, expect, it } from "vitest";
import { getModel, streamSimple } from "../src/compat.ts";
import { findEnvKeys, getEnvApiKey } from "../src/env-api-keys.ts";
import { builtinModels, getBuiltinClassifierModels } from "../src/providers/all.ts";
import type { ClassifierContext } from "../src/types.ts";

const originalBasetenApiKey = process.env.BASETEN_API_KEY;

afterEach(() => {
	if (originalBasetenApiKey === undefined) {
		delete process.env.BASETEN_API_KEY;
	} else {
		process.env.BASETEN_API_KEY = originalBasetenApiKey;
	}
});

describe("Baseten models", () => {
	it("keeps both GLM 5.2 endpoints text-only", () => {
		expect(getModel("baseten", "zai-org/GLM-5.2").input).toEqual(["text"]);
		expect(getModel("baseten", "zai-org/GLM-5.2-Fast").input).toEqual(["text"]);
	});

	it("sends Baseten chat_template_args with reasoning effort", async () => {
		const model = getModel("baseten", "zai-org/GLM-5.2");
		let payload: Record<string, unknown> | undefined;

		await streamSimple(
			model,
			{ messages: [{ role: "user", content: "test", timestamp: 0 }] },
			{
				apiKey: "test-baseten-key",
				reasoning: "high",
				onPayload: (value) => {
					payload = value as Record<string, unknown>;
					throw new Error("payload captured");
				},
			},
		).result();

		expect(payload?.chat_template_args).toEqual({ enable_thinking: true });
		expect(payload?.reasoning_effort).toBe("high");
	});

	it("disables Baseten opt-in reasoning when thinking is off", async () => {
		const model = getModel("baseten", "zai-org/GLM-5.2");
		let payload: Record<string, unknown> | undefined;

		await streamSimple(
			model,
			{ messages: [{ role: "user", content: "test", timestamp: 0 }] },
			{
				apiKey: "test-baseten-key",
				onPayload: (value) => {
					payload = value as Record<string, unknown>;
					throw new Error("payload captured");
				},
			},
		).result();

		expect(payload?.chat_template_args).toEqual({ enable_thinking: false });
		expect(payload?.reasoning_effort).toBe("none");
	});

	it("resolves BASETEN_API_KEY from the environment", () => {
		process.env.BASETEN_API_KEY = "test-baseten-key";

		expect(findEnvKeys("baseten")).toEqual(["BASETEN_API_KEY"]);
		expect(getEnvApiKey("baseten")).toBe("test-baseten-key");
	});

	it("lists the decision models as text-only System One classifiers", () => {
		expect(
			getBuiltinClassifierModels("baseten").map(({ id, api, baseUrl, input, contextWindow }) => ({
				id,
				api,
				baseUrl,
				input,
				contextWindow,
			})),
		).toEqual(
			["decision-models/baseten-elephant-9b-latest", "decision-models/baseten-mammoth-27b-latest"].map((id) => ({
				id,
				api: "typesafe-system-one",
				baseUrl: "https://inference.baseten.co/v1",
				input: ["text"],
				contextWindow: 16384,
			})),
		);
	});

	it("posts classify requests to Baseten's System One endpoint", async () => {
		const models = builtinModels();
		const elephant = models.getModelOfType("classifier", "baseten", "decision-models/baseten-elephant-9b-latest");
		if (!elephant) throw new Error("missing Baseten Elephant model");
		const context: ClassifierContext = {
			state: { text: "Ship it." },
			questions: {
				approved: {
					type: "bool",
					instructions: "Does the message approve the change?",
					criteria: { true: "Approval", false: "No approval" },
				},
			},
		};
		const requests: Array<{ url: string; body: unknown; headers: Headers }> = [];

		const result = await models.classify(elephant, context, {
			apiKey: "test-baseten-key",
			fetch: async (input, init) => {
				requests.push({
					url: String(input),
					body: JSON.parse(String(init?.body)),
					headers: new Headers(init?.headers),
				});
				return Response.json({ answers: { approved: { type: "noul", noul: 0.9 } } });
			},
		});

		expect(requests).toHaveLength(1);
		expect(requests[0]?.url).toBe("https://inference.baseten.co/v1/systemone");
		expect(requests[0]?.body).toEqual({
			model: "decision-models/baseten-elephant-9b-latest",
			state: context.state,
			questions: { approved: { ...context.questions.approved, type: "noul" } },
		});
		expect(requests[0]?.headers.get("authorization")).toBe("Bearer test-baseten-key");
		expect(requests[0]?.headers.has("TypeSafe-Danger-Zone-Images")).toBe(false);
		expect(result.stopReason).toBe("stop");
		expect(result.answers.approved).toEqual({ type: "bool", probability: 0.9 });
	});
});
