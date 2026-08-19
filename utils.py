import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses,
    val_losses,
    save_path="loss_curve.png"
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
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.show()