"""
example_mlp.py — Training a tiny MLP with gradient descent.

Task: binary classification on four 2-D points.
Run: PYTHONPATH=. python3 examples/example_mlp.py
"""

from micrograd.nn import MLP


def main() -> None:
    print("=" * 55)
    print("  MiniGrad — MLP Training Demo")
    print("=" * 55)

    X = [[2.0, 3.0], [-1.0, -1.0], [-2.0, 1.5], [1.0, -2.0]]
    y_true = [1.0, -1.0, -1.0, 1.0]

    model = MLP(2, [4, 4, 1])
    print(f"\nModel: {model}")
    print(f"Parameters: {len(model.parameters())}\n")

    lr = 0.05
    print(f"{'Epoch':>6}  {'Loss':>12}")
    print("-" * 22)

    for epoch in range(1, 21):
        y_pred = [model(x) for x in X]
        losses = [(1.0 - yi * pi) ** 2 for yi, pi in zip(y_true, y_pred)]
        total_loss = sum(losses[1:], losses[0])

        model.zero_grad()
        total_loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch == 1 or epoch % 5 == 0:
            print(f"{epoch:>6}  {total_loss.data:>12.6f}")

    print("\nFinal predictions:")
    for xi, yi in zip(X, y_true):
        pred = model(xi)
        correct = "✓" if (pred.data > 0) == (yi > 0) else "✗"
        print(
            f"  x={[round(v, 1) for v in xi]}  "
            f"pred={pred.data:+.4f}  true={yi:+.0f}  {correct}"
        )


if __name__ == "__main__":
    main()
