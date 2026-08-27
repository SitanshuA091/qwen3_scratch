import os
import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses,
    val_losses,
    save_dir,
    model_type
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type} Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    save_path = os.path.join(
        save_dir,
        f"{model_type}_loss_curve.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot -> {save_path}")
