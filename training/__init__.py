# EmberNet Training
# Import only what's needed to avoid cascade import issues

__all__ = [
    "EmberNetDataset",
    "create_dataloaders",
    "DataConfig",
    "Trainer",
    "TrainingConfig"
]

def __getattr__(name):
    """Lazy imports to avoid loading all dependencies at once."""
    if name in ("EmberNetDataset", "create_dataloaders", "DataConfig"):
        from .data import EmberNetDataset, create_dataloaders, DataConfig
        return {"EmberNetDataset": EmberNetDataset,
                "create_dataloaders": create_dataloaders,
                "DataConfig": DataConfig}[name]
    elif name in ("Trainer", "TrainingConfig"):
        from .train import Trainer, TrainingConfig
        return {"Trainer": Trainer, "TrainingConfig": TrainingConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

