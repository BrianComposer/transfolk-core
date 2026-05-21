from __future__ import annotations

from tests_tfcore.common import assert_equal, assert_true, check, require_module


def run_tests(ctx):
    def loss_factory_builds_cross_entropy_ignore_pad():
        torch = require_module("torch")
        from transfolk_core.config.entities.runtime_train import RuntimeTrain
        from transfolk_core.training.loss_factory import LossFactory
        loss = LossFactory.build(RuntimeTrain(id=1, name="rt", loss="cross_entropy"))
        assert_true(isinstance(loss, torch.nn.CrossEntropyLoss), "Debe crear torch.nn.CrossEntropyLoss.")
        assert_equal(loss.ignore_index, 0)

    check(ctx, "LossFactory: cross_entropy con ignore_index=0", loss_factory_builds_cross_entropy_ignore_pad)

    def optimizer_factory_builds_supported_optimizers():
        torch = require_module("torch")
        from transfolk_core.config.entities.runtime_train import RuntimeTrain
        from transfolk_core.training.optimizer_factory import OptimizerFactory
        model = torch.nn.Linear(2, 1)
        for opt_name, expected_cls in [("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)]:
            opt = OptimizerFactory.build(RuntimeTrain(id=1, name="rt", optimizer=opt_name, learning_rate=1e-3), model)
            assert_true(isinstance(opt, expected_cls), f"{opt_name} debe crear {expected_cls}")

    check(ctx, "OptimizerFactory: adamw/adam/sgd", optimizer_factory_builds_supported_optimizers)

    def scheduler_factory_builds_none_linear_cosine():
        torch = require_module("torch")
        from transfolk_core.config.entities.runtime_train import RuntimeTrain
        from transfolk_core.training.scheduler_factory import SchedulerFactory
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        assert_equal(SchedulerFactory.build(RuntimeTrain(id=1, name="rt", scheduler="none"), opt, total_steps=10), None)
        linear = SchedulerFactory.build(RuntimeTrain(id=1, name="rt", scheduler="linear", warmup_steps=2), opt, total_steps=10)
        cosine = SchedulerFactory.build(RuntimeTrain(id=1, name="rt", scheduler="cosine", warmup_steps=2), opt, total_steps=10)
        assert_true(linear is not None, "Scheduler linear no debe ser None.")
        assert_true(cosine is not None, "Scheduler cosine no debe ser None.")

    check(ctx, "SchedulerFactory: none/linear/cosine", scheduler_factory_builds_none_linear_cosine)
