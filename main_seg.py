import torch

from config.train_setup import TrainSetup
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions
from util.engineSeg import optimize_model, test_Predict_Rank, validate_model


def main():
    opt, _, _, _, plots = TrainOptions().parse()

    data_loader = CreateDataLoader(opt)
    train_loader, val_loader, test_loader, datalen = data_loader.load_data()
    print("#Data loader scheme created")

    model = create_model(opt)
    train_config = TrainSetup(opt, model)
    model = train_config.Config.model

    for epoch in range(train_config.Config.tracking_metrics["start_epoch"], opt.epochs):
        print("-" * 10, flush=True)
        train_config.Config.tracking_metrics["epoch"] = epoch

        (
            model,
            train_config.Config.optimizer,
            train_loader,
            train_config.Config.tracking_metrics,
            opt.wandb_logger,
        ) = optimize_model(
            model=model,
            optimizer=train_config.Config.optimizer,
            loss_func=train_config.Config.loss_function,
            scaler=train_config.Config.loss_scaler,
            train_gen=train_loader,
            args=opt,
            tracking_metrics=train_config.Config.tracking_metrics,
            wandb_logger=opt.wandb_logger,
            Config=train_config,
            Debug=plots if opt.debug else None,
        )

        if opt.sched is not None:
            if opt.sched in ("warmup_cosine", "cosine_anneal"):
                train_config.Config.lr_scheduler.step()
                lr_update = train_config.Config.optimizer.param_groups[0]["lr"]
            elif opt.sched == "poly":
                lr_update = train_config.Config.lr_scheduler.step(epoch + 1)
                train_config.Config.optimizer.param_groups[0]["lr"] = lr_update
            else:
                lr_update = train_config.Config.optimizer.param_groups[0]["lr"]
            print(f"Learning Rate Updated! New Value: {lr_update:.10}", flush=True)
        else:
            lr_update = train_config.Config.optimizer.param_groups[0]["lr"]
            print(f"Learning Rate fixed: {lr_update:.10}", flush=True)

        if opt.enable_wandb:
            opt.wandb_logger.log({"lr/epoch": lr_update}, step=epoch)

        should_validate = (epoch + 1) % opt.val_interval == 0
        should_validate &= epoch + 1 >= opt.validate_min_epoch
        if should_validate:
            model.eval()
            with torch.no_grad():
                (
                    model,
                    train_config.Config.optimizer,
                    val_loader,
                    train_config.Config.tracking_metrics,
                    opt.wandb_logger,
                ) = validate_model(
                    model=model,
                    loss_func=train_config.Config.loss_function,
                    optimizer=train_config.Config.optimizer,
                    valid_gen=val_loader,
                    args=opt,
                    tracking_metrics=train_config.Config.tracking_metrics,
                    wandb_logger=opt.wandb_logger,
                    Config=train_config,
                )
            plots.save_Loss_Metrics(
                train_config.Config.tracking_metrics["all_train_loss"],
                train_config.Config.tracking_metrics["all_valid_loss"],
                train_config.Config.tracking_metrics["all_valid_metrics_Dice"],
                opt.val_interval,
            )

    print(
        "Training Complete! Peak Validation Ranking Score: "
        f"{train_config.Config.tracking_metrics['best_metric']:.4f} "
        f"@ Epoch: {train_config.Config.tracking_metrics['best_metric_epoch']}"
    )
    test_Predict_Rank(model, opt, test_loader, datalen[1])

    if opt.enable_wandb:
        opt.wandb_logger.finish()


if __name__ == "__main__":
    main()
