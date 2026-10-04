"""Tkinter interface for configuring and training the MNIST digit network."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from dnn import DigitNetwork, FILES, accuracy, load_images, load_labels, one_hot


class NetworkGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Network")
        self.root.geometry("760x760")
        self.root.minsize(640, 600)
        self.events: queue.Queue[tuple] = queue.Queue()
        self.layer_rows: list[tuple[tk.IntVar, tk.StringVar, ttk.Frame]] = []

        style = ttk.Style()
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Section.TLabel", font=("TkDefaultFont", 11, "bold"))
        style.configure("Train.TButton", font=("TkDefaultFont", 11, "bold"), padding=8)

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Handwritten Digit Network", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Configure the network, then train it on the MNIST handwritten digit dataset.",
        ).pack(anchor="w", pady=(4, 14))

        arch = ttk.LabelFrame(outer, text="Network architecture", padding=12)
        arch.pack(fill="x")
        ttk.Label(arch, text="Input", width=16).grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(arch, text="784 values (28 x 28 pixels)").grid(row=0, column=1, sticky="w")

        self.rows_frame = ttk.Frame(arch)
        self.rows_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        ttk.Label(arch, text="Output", width=16).grid(row=2, column=0, sticky="w", pady=3)
        ttk.Label(arch, text="10 digits (0-9)").grid(row=2, column=1, sticky="w")
        ttk.Label(arch, text="Activation:").grid(row=2, column=2, sticky="e", padx=(12, 4))
        self.output_activation = tk.StringVar(value="sigmoid")
        ttk.Combobox(
            arch,
            textvariable=self.output_activation,
            values=("relu", "sigmoid"),
            state="readonly",
            width=10,
        ).grid(row=2, column=3, sticky="w")
        arch.columnconfigure(1, weight=1)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(9, 16))
        ttk.Button(controls, text="+ Add hidden layer", command=self.add_layer).pack(side="left")
        ttk.Button(controls, text="− Remove last layer", command=self.remove_layer).pack(
            side="left", padx=8
        )

        settings = ttk.LabelFrame(outer, text="Training settings", padding=12)
        settings.pack(fill="x")
        self.epochs = self._setting(settings, "Epochs", "8", 0)
        self.batch_size = self._setting(settings, "Batch size", "64", 1)
        self.learning_rate = self._setting(settings, "Learning rate", "0.05", 2)

        self.train_button = ttk.Button(
            outer, text="Train network", style="Train.TButton", command=self.start_training
        )
        self.train_button.pack(fill="x", pady=(14, 8))
        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status = tk.StringVar(value="Ready. Default network: 784 → 128 → 64 → 10")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(6, 8))

        log_frame = ttk.LabelFrame(outer, text="Training progress", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=9, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

        self.add_layer(128, "relu")
        self.add_layer(64, "relu")
        self.root.after(100, self.process_events)

    @staticmethod
    def _setting(parent: ttk.Frame, label: str, default: str, row: int) -> tk.StringVar:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        value = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=value, width=14).grid(row=row, column=1, sticky="w", padx=8)
        return value

    def add_layer(self, nodes: int = 64, activation: str = "relu") -> None:
        index = len(self.layer_rows) + 1
        row = ttk.Frame(self.rows_frame)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=f"Hidden layer {index}", width=16).pack(side="left")
        node_count = tk.IntVar(value=nodes)
        ttk.Spinbox(row, from_=1, to=2048, textvariable=node_count, width=9).pack(side="left")
        ttk.Label(row, text="nodes   activation:").pack(side="left", padx=(8, 4))
        activation_value = tk.StringVar(value=activation)
        ttk.Combobox(
            row,
            textvariable=activation_value,
            values=("relu", "sigmoid"),
            state="readonly",
            width=10,
        ).pack(side="left")
        self.layer_rows.append((node_count, activation_value, row))
        self._renumber_layers()

    def remove_layer(self) -> None:
        if not self.layer_rows:
            return
        _, _, row = self.layer_rows.pop()
        row.destroy()
        self._renumber_layers()

    def _renumber_layers(self) -> None:
        for index, (_, _, row) in enumerate(self.layer_rows, start=1):
            row.winfo_children()[0].configure(text=f"Hidden layer {index}")

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start_training(self) -> None:
        try:
            hidden_sizes = [int(nodes.get()) for nodes, _, _ in self.layer_rows]
            if any(nodes < 1 or nodes > 2048 for nodes in hidden_sizes):
                raise ValueError("Each hidden layer must have between 1 and 2048 nodes.")
            epochs = int(self.epochs.get())
            batch_size = int(self.batch_size.get())
            learning_rate = float(self.learning_rate.get())
            if epochs < 1 or batch_size < 1 or learning_rate <= 0:
                raise ValueError("Epochs, batch size, and learning rate must be positive.")
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Check training settings", str(error), parent=self.root)
            return

        layer_sizes = [784, *hidden_sizes, 10]
        activations = [value.get() for _, value, _ in self.layer_rows]
        activations.append(self.output_activation.get())
        config = (layer_sizes, activations, epochs, batch_size, learning_rate)
        self.train_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=epochs)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status.set("Preparing training data...")
        threading.Thread(target=self._train_worker, args=config, daemon=True).start()

    def _train_worker(
        self,
        layer_sizes: list[int],
        activations: list[str],
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        try:
            self.events.put(("status", "Loading MNIST data (first run downloads it)..."))
            train_x = load_images(FILES["train_images"])
            train_labels = load_labels(FILES["train_labels"])
            test_x = load_images(FILES["test_images"])
            test_labels = load_labels(FILES["test_labels"])
            train_y = one_hot(train_labels)
            model = DigitNetwork(layer_sizes, activations)
            rng = np.random.default_rng(42)

            for epoch in range(epochs):
                order = rng.permutation(len(train_x))
                losses = []
                for start in range(0, len(order), batch_size):
                    indices = order[start : start + batch_size]
                    losses.append(model.train_batch(train_x[indices], train_y[indices], learning_rate))
                score = accuracy(model, test_x, test_labels)
                self.events.put(("epoch", epoch + 1, float(np.mean(losses)), score))

            model_path = Path(__file__).with_name("digit_model.npz")
            model.save(model_path)
            self.events.put(("complete", str(model_path), layer_sizes, activations))
        except Exception as error:
            self.events.put(("error", str(error)))

    def process_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "status":
                    self.status.set(event[1])
                elif event[0] == "epoch":
                    _, epoch, loss, score = event
                    self.progress.configure(value=epoch)
                    message = f"Epoch {epoch}: loss {loss:.4f} | test accuracy {score:.2%}"
                    self.status.set(message)
                    self._append_log(message)
                elif event[0] == "complete":
                    _, path, layer_sizes, activations = event
                    self.status.set(f"Training complete. Model saved to {path}")
                    self._append_log(f"Saved model: {path}")
                    self._append_log(
                        "Architecture: "
                        + " → ".join(str(size) for size in layer_sizes)
                        + " | activations: "
                        + ", ".join(activations)
                    )
                    self.train_button.configure(state="normal")
                elif event[0] == "error":
                    self.status.set("Training stopped with an error.")
                    self._append_log("Error: " + event[1])
                    self.train_button.configure(state="normal")
                    messagebox.showerror("Training error", event[1], parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self.process_events)


def launch() -> None:
    root = tk.Tk()
    NetworkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
