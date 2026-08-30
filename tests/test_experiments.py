import unittest
from dataclasses import replace
from datetime import timedelta

from agents import (
    BatchExperimentRunner,
    ExperimentConfigurationError,
    ExperimentVariant,
)
from tests.test_backtest import run


def baseline():
    return ExperimentVariant(
        name="baseline",
        agent_versions=("news_analyst:v1", "company_analyst:v1", "risk_llm:v1"),
        strategy_config={"max_position_pct": 0.10, "min_evidence_count": 2},
        model_labels={"news_analyst": "model-a"},
        prompt_labels={"news_analyst": "prompt-v1"},
    )


class BatchExperimentRunnerTests(unittest.TestCase):
    def test_builds_agent_and_threshold_ablations(self):
        base = baseline()
        without_news = base.without_agent("news_analyst")
        stricter = base.with_strategy("strict_governance", min_evidence_count=3)
        model_test = base.with_models("new_model", news_analyst="model-b")
        prompt_test = base.with_prompts("new_prompt", news_analyst="prompt-v2")
        self.assertNotIn("news_analyst:v1", without_news.agent_versions)
        self.assertNotIn("news_analyst", without_news.model_labels)
        self.assertEqual(stricter.strategy_config["min_evidence_count"], 3)
        self.assertEqual(base.strategy_config["min_evidence_count"], 2)
        self.assertEqual(model_test.model_labels["news_analyst"], "model-b")
        self.assertEqual(prompt_test.prompt_labels["news_analyst"], "prompt-v2")

    def test_runs_matrix_and_computes_metric_deltas(self):
        base = ExperimentVariant(
            name="baseline",
            agent_versions=("trade_constructor:v1",),
            strategy_config={"max_position_pct": 0.10},
        )
        smaller = base.with_strategy("smaller_position", max_position_pct=0.05)

        def execute(variant):
            config = {
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                **variant.strategy_config,
            }
            return run(
                agent_versions=variant.agent_versions,
                strategy_config=config,
            )

        report = BatchExperimentRunner(execute).run((base, smaller))
        self.assertEqual(len(report.succeeded), 2)
        self.assertEqual(report.failed, ())
        self.assertEqual(report.result("baseline").metric_deltas["total_return"], 0)
        self.assertLess(
            report.result("smaller_position").metric_deltas["total_return"], 0
        )

    def test_failed_non_baseline_variant_is_reported_without_losing_results(self):
        base = ExperimentVariant(
            name="baseline", agent_versions=("trade_constructor:v1",),
            strategy_config={},
        )
        broken = replace(base, name="broken")

        def execute(variant):
            if variant.name == "broken":
                raise RuntimeError("synthetic failure")
            return run()

        report = BatchExperimentRunner(execute).run((base, broken))
        self.assertEqual(len(report.succeeded), 1)
        self.assertEqual(report.failed[0].error_type, "RuntimeError")

    def test_rejects_duplicate_names_and_failed_baseline(self):
        base = ExperimentVariant(
            name="baseline", agent_versions=("trade_constructor:v1",),
            strategy_config={},
        )
        with self.assertRaises(ExperimentConfigurationError):
            BatchExperimentRunner(lambda variant: run()).run((base, base))
        with self.assertRaisesRegex(ExperimentConfigurationError, "baseline.*failed"):
            BatchExperimentRunner(
                lambda variant: (_ for _ in ()).throw(RuntimeError("failed"))
            ).run((base,))

    def test_cannot_remove_an_agent_that_is_not_present(self):
        with self.assertRaises(ExperimentConfigurationError):
            baseline().without_agent("portfolio_risk")


if __name__ == "__main__":
    unittest.main()
