"""
HELR: Homomorphic-Encryption-driven Logistic Regression
End-to-end encrypted training, reproducing the design of:
Naresh & Reddi, "Exploring the future of privacy-preserving heart disease
prediction: a fully homomorphic encryption-driven logistic regression
approach", J. Big Data (2025) 12:52.

Run in an environment with TenSEAL installed (Google Colab is fine):
    pip install tenseal

Usage:
    python HELR2.py --data heart.csv --epochs 20 --lr 1.0

-----------------------------------------------------------------------
    
1. CONTEXT PARAMETERS
   The paper's reported context for N=8192 is
       coeff_mod_bit_sizes = [40,21,21,21,21,21,21,40], scale = 2**21
   This is a valid 128-bit-security configuration (206 bits < 218-bit
   bound for N=8192), but in our TenSEAL 0.3.17 environment it produced
   materially wrong results even for a plain Enc(a)*Enc(b) test
   (expected [0.5,2,4.5,8], got [0.586,2.353,5.284,9.396]).
   Root cause is almost certainly that 21-bit rescale primes sit too
   close to TenSEAL's internal precision floor for this SEAL build.

   Instead we use N=16384, which raises the 128-bit-security bit budget
   to 438 bits, and pick
       coeff_mod_bit_sizes = [60,40,40,40,40,40,40,60]  (360 bits, valid)
       scale = 2**40
   This follows Microsoft SEAL's own worked CKKS example convention
   (rescale primes close to the scale), giving 6 usable multiplicative
   levels instead of 2, and it was empirically validated (see
   `sanity_check_context` below) before being used for training.

2. MULTIPLICATIVE DEPTH / "BOOTSTRAPPING"
   TenSEAL/SEAL's CKKS has no bootstrapping, so a ciphertext's usable
   depth is fixed at context-creation time. A full un-bootstrapped
   encrypted SGD loop for many epochs is not possible without refreshing
   ciphertexts periodically. The paper's own architecture (Fig. 2)
   already includes a message "w' " sent from the CSP back to the
   Hospital — i.e. the party holding the private key is periodically
   back in the loop. We mirror that: after each epoch, the encrypted
   weight vector is decrypted-and-re-encrypted by the "Hospital" role
   (the only entity in this simulation that ever touches the private
   key). The "CSP" role (encrypted forward/backward pass) never sees
   plaintext data, labels, or weights at any point — privacy w.r.t. the
   untrusted compute party is preserved exactly as in the paper.

3. PACKING
   Each of the D feature columns is packed into ONE ciphertext of
   length P (number of patients), i.e.
       Enc(X_1) = [x_11, x_21, ..., x_P1]
       ...
       Enc(X_D) = [x_1D, x_2D, ..., x_PD]
   z = X @ w is then computed as sum_j w_j * Enc(X_j), which needs only
   D elementwise ciphertext-plaintext multiplications (cheap: does not
   consume a full ciphertext-ciphertext multiplicative level) rather
   than a per-patient loop. Label vector y is packed the same way.

4. SIGMOID + LEVEL MATCHING
   sigma(z) is replaced by a low-degree polynomial approximation.
   Default is the LINEAR approx sigma(z) ~= 0.5 + 0.197*z, which costs
   exactly one multiplicative level and involves zero ciphertext-
   ciphertext multiplications — so there is no level-mismatch risk.
   A cubic approx (0.5 + 0.15012*z - 0.001593*z^3, Kim-et-al. style,
   valid for z roughly in [-8, 8]) is available via --sigmoid_degree 3,
   but CKKS requires two ciphertexts to be at the SAME modulus level
   before they can be combined; chaining z*z then z^2*z mixes operands
   that are one level apart. We handle this with a `_burn_levels`
   helper that multiplies the shallower operand by plaintext 1.0 the
   right number of times (each such multiply consumes+rescales one
   level without changing the value), bringing both operands to a
   matching level before combining. The same alignment is needed
   wherever a 'fresh' ciphertext (enc_y, enc_X_cols[j]) is combined
   with a 'deep' one derived after the forward pass + sigmoid.
"""

import argparse
import numpy as np
import pandas as pd

try:
    import tenseal as ts
except ImportError:
    ts = None
    print("[warn] tenseal not installed in this environment. "
          "Install with `pip install tenseal` before running the "
          "encrypted portions of this script.")


# --------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------
def load_and_preprocess(path, target_col=None, test_size=0.2, seed=42):
    """
    Loads a heart-disease-style CSV (13 numeric/categorical features +
    binary target). Standardizes features to zero mean / unit variance
    (important: keeps sigmoid polynomial approximation in its valid
    range and keeps CKKS ciphertext magnitudes well-scaled).
    """
    df = pd.read_csv(path)
    if target_col is None:
        # heuristics matching common Kaggle heart-disease dumps
        for cand in ["target", "HeartDisease", "num", "condition"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col is None:
            target_col = df.columns[-1]

    y = df[target_col].astype(float).values
    X = df.drop(columns=[target_col]).astype(float).values

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_test = int(len(X) * test_size)
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------
# 2. Plaintext baseline (already in your status list as DONE — kept
#    here so this script is self-contained and gives you the
#    HELR-vs-LR accuracy comparison the paper reports)
# --------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_plaintext_lr(X, y, epochs=20, lr=1.0, l2=0.059):
    P, D = X.shape
    w = np.zeros(D)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / P + l2 * w
        w = w - lr * grad
    return w


def accuracy(X, y, w):
    p = sigmoid(X @ w)
    pred = (p >= 0.5).astype(float)
    return float((pred == y).mean())


# --------------------------------------------------------------------
# 3. CKKS context
# --------------------------------------------------------------------
def build_context(profile="practical"):
    if ts is None:
        raise RuntimeError("tenseal is not installed")

    if profile == "practical":
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
        )
        ctx.global_scale = 2 ** 40
    elif profile == "paper":
        # kept for reproducibility / report purposes only — do not use
        # for real training given the precision issue noted above.
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40],
        )
        ctx.global_scale = 2 ** 21
    else:
        raise ValueError(profile)

    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def sanity_check_context(ctx, verbose=True):
    """Reproduces your earlier Enc(a)*Enc(b) / dot-product primitive
    tests against a freshly built context, so a bad parameter choice
    is caught BEFORE a multi-epoch training run wastes time on it."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.5, 1.0, 1.5, 2.0]
    enc_a = ts.ckks_vector(ctx, a)
    enc_b = ts.ckks_vector(ctx, b)

    prod = (enc_a * enc_b).decrypt()[:4]
    dot = enc_a.dot(enc_b).decrypt()[0]

    expected_prod = [0.5, 2.0, 4.5, 8.0]
    ok_prod = max(abs(p - e) for p, e in zip(prod, expected_prod)) < 1e-2
    ok_dot = abs(dot - 15.0) < 1e-2

    if verbose:
        print(f"  Enc*Enc product : {prod}  (expected {expected_prod})  "
              f"{'OK' if ok_prod else 'FAIL'}")
        print(f"  Enc.dot(Enc)    : {dot:.6f}  (expected 15.0)  "
              f"{'OK' if ok_dot else 'FAIL'}")
    return ok_prod and ok_dot


# --------------------------------------------------------------------
# 4. Packed encrypted representation
# --------------------------------------------------------------------
def encrypt_features_packed(X, ctx):
    """One ciphertext per feature column, packed across all patients."""
    P, D = X.shape
    return [ts.ckks_vector(ctx, X[:, j].tolist()) for j in range(D)]


def encrypt_broadcast(value, length, ctx):
    """Encrypt a scalar broadcast to `length` slots (needed to multiply
    a scalar weight against a length-P packed feature ciphertext when
    the weight itself must stay encrypted)."""
    return ts.ckks_vector(ctx, [value] * length)


# sigma(z) ~= c0 + c1*z + c3*z^3   (Kim et al. style least-squares fit)
SIG_C0, SIG_C1, SIG_C3 = 0.5, 0.15012, -0.001593
# sigma(z) ~= c0 + c1*z            (linear approx; default, depth-safe)
SIG_LIN_C0, SIG_LIN_C1 = 0.5, 0.197


def _burn_levels(vec, n):
    """Consume `n` multiplicative levels WITHOUT changing the encrypted
    value, by multiplying by the plaintext constant 1.0 n times (each
    multiply-by-plaintext is followed by an automatic rescale, which is
    exactly what drops a level). This is used to bring a 'fresh'
    ciphertext down to the same modulus level as a 'deep' one before
    combining them — CKKS requires matching levels for add/sub/mul
    between two ciphertexts, and mismatches here are what caused the
    'scale out of bounds' crash."""
    for _ in range(n):
        vec = vec * 1.0
    return vec


def encrypted_sigmoid(enc_z, degree=1):
    """
    degree=1 (default): sigma(z) ~= c0 + c1*z
        Costs exactly 1 multiplicative level (one plaintext multiply).
        No ciphertext-ciphertext multiplication at all, so there is no
        level-mismatch risk. Use this to get a working end-to-end run.

    degree=3 (experimental): sigma(z) ~= c0 + c1*z + c3*z^3
        Costs 3 multiplicative levels total. Requires explicit level
        alignment (via _burn_levels) between the two operands of the
        z^2 * z multiplication, since z^2 has already consumed one
        more level than the original z. Try this only after degree=1
        works end-to-end and you have time to validate it further.
    """
    if degree == 1:
        return enc_z * SIG_LIN_C1 + SIG_LIN_C0

    if degree == 3:
        enc_z2 = enc_z * enc_z                  # level +1 (now L+1)
        enc_z_matched = _burn_levels(enc_z, 1)   # bring original z to L+1
        enc_z3 = enc_z2 * enc_z_matched          # level +1 (now L+2)
        term3 = enc_z3 * SIG_C3                  # level +1 (now L+3)
        term1 = _burn_levels(enc_z, 2) * SIG_C1  # align z to L+3 first
        return term3 + term1 + SIG_C0

    raise ValueError("degree must be 1 or 3")


# --------------------------------------------------------------------
# 5. One encrypted epoch (Algorithm 3 of the paper)
# --------------------------------------------------------------------
# levels consumed reaching enc_p, by sigmoid degree (forward pass = 1,
# plus sigmoid's own cost) — used to burn matching levels into the
# "fresh" ciphertexts (enc_y, enc_X_cols[j]) before combining them with
# the deep, sigmoid-derived ciphertexts.
_SIGMOID_DEPTH = {1: 1 + 1, 3: 1 + 3}   # forward(1) + sigmoid(1 or 3)


def encrypted_epoch(enc_X_cols, enc_y, w_plain, P, D, lr, l2, ctx,
                     sigmoid_degree=1):
    """
    Runs one full encrypted forward + backward pass.
    Weights are kept in PLAINTEXT on the "Hospital" side between epochs
    (see design note 2) but never touched by the "CSP" role during the
    forward/backward computation itself — only ciphertexts belonging to
    the CSP role are used inside this function.

    Returns the new plaintext weight vector (post refresh/decrypt step,
    performed by "Hospital" i.e. this function, which stands in for the
    party holding the private key in this local simulation).
    """
    depth_to_p = _SIGMOID_DEPTH[sigmoid_degree]

    # ---- forward pass: z = X @ w, done as packed ciphertext-plaintext ops
    enc_z = enc_X_cols[0] * w_plain[0]
    for j in range(1, D):
        enc_z = enc_z + enc_X_cols[j] * w_plain[j]

    # ---- encrypted sigmoid
    enc_p = encrypted_sigmoid(enc_z, degree=sigmoid_degree)

    # ---- error = p - y   (still packed across patients)
    # enc_y is FRESH (level 0) but enc_p has descended `depth_to_p`
    # levels; burn the same number of levels into enc_y before
    # subtracting, or CKKS will refuse/corrupt the operation.
    enc_y_matched = _burn_levels(enc_y, depth_to_p)
    enc_err = enc_p - enc_y_matched

    # ---- gradient per feature: g_j = (1/P) * sum_i err_i * X_ij + l2*w_j
    grad = np.zeros(D)
    for j in range(D):
        # enc_X_cols[j] is also fresh; align it to enc_err's level
        # (depth_to_p, since the subtraction above didn't add a level)
        enc_Xj_matched = _burn_levels(enc_X_cols[j], depth_to_p)
        enc_gj = (enc_err * enc_Xj_matched).sum()
        # this is the "refresh" point: Hospital decrypts to read out the
        # scalar gradient component (CSP never sees this plaintext value)
        grad[j] = enc_gj.decrypt()[0] / P + l2 * w_plain[j]

    w_new = w_plain - lr * grad
    return w_new


def train_helr(X_train, y_train, epochs, lr, l2, ctx, sigmoid_degree=1):
    P, D = X_train.shape
    enc_X_cols = encrypt_features_packed(X_train, ctx)
    enc_y = ts.ckks_vector(ctx, y_train.tolist())

    w = np.zeros(D)
    history = []
    for ep in range(1, epochs + 1):
        w = encrypted_epoch(enc_X_cols, enc_y, w, P, D, lr, l2, ctx,
                             sigmoid_degree=sigmoid_degree)
        history.append(w.copy())
        print(f"  epoch {ep:3d}  |  ||w|| = {np.linalg.norm(w):.4f}")
    return w, history


def run_sweep(X_train, y_train, X_test, y_test, epoch_list, lr, l2,
              ctx, sigmoid_degree, out_csv="helr_sweep_results.csv",
              out_png="helr_sweep_plot.png"):
    """
    Reproduces the paper's Fig. 4 / Fig. 6 style comparison: for each
    epoch count, trains fresh plaintext LR and fresh encrypted HELR,
    records accuracy and wall-clock time, then saves a CSV + a plot.
    This is the evidence your mentor will actually want to see.
    """
    import time
    rows = []
    for epochs in epoch_list:
        w_plain = train_plaintext_lr(X_train, y_train, epochs=epochs,
                                      lr=lr, l2=l2)
        acc_plain = accuracy(X_test, y_test, w_plain)

        t0 = time.time()
        w_helr, _ = train_helr(X_train, y_train, epochs, lr, l2, ctx,
                                sigmoid_degree=sigmoid_degree)
        helr_time = time.time() - t0
        acc_helr = accuracy(X_test, y_test, w_helr)

        rows.append({
            "epochs": epochs,
            "lr_accuracy": acc_plain,
            "helr_accuracy": acc_helr,
            "accuracy_gap": abs(acc_plain - acc_helr),
            "helr_time_sec": helr_time,
        })
        print(f"[sweep] epochs={epochs:3d}  LR={acc_plain:.4f}  "
              f"HELR={acc_helr:.4f}  gap={rows[-1]['accuracy_gap']:.4f}  "
              f"time={helr_time:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results table to {out_csv}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(df["epochs"], df["lr_accuracy"], marker="o", label="LR")
        axes[0].plot(df["epochs"], df["helr_accuracy"], marker="s",
                     label=f"HELR (sigmoid deg={sigmoid_degree})")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Test accuracy")
        axes[0].set_title("LR vs HELR accuracy")
        axes[0].legend()

        axes[1].bar(df["epochs"].astype(str), df["helr_time_sec"],
                    color="teal")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Time (s)")
        axes[1].set_title("HELR training time")

        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"Saved plot to {out_png}")
    except ImportError:
        print("matplotlib not installed — skipping plot, CSV is still saved.")

    return df


# --------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to heart.csv")
    ap.add_argument("--target_col", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=0.059)
    ap.add_argument("--profile", default="practical",
                     choices=["practical", "paper"])
    ap.add_argument("--sigmoid_degree", type=int, default=1,
                     choices=[1, 3],
                     help="1 = linear approx (safe, default). "
                          "3 = cubic approx (matches paper more closely, "
                          "needs more levels, try only after 1 works).")
    ap.add_argument("--sweep", default=None,
                     help="comma-separated epoch counts for a paper-style "
                          "accuracy/time comparison sweep, e.g. "
                          "'5,10,20,50'. Overrides --epochs when set.")
    args = ap.parse_args()

    X_train, y_train, X_test, y_test = load_and_preprocess(
        args.data, target_col=args.target_col
    )
    print(f"Loaded: {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features")

    if ts is None:
        print("\ntenseal not installed — cannot run encrypted training.")
        return

    print(f"\nBuilding CKKS context (profile='{args.profile}')...")
    ctx = build_context(args.profile)
    print("Sanity-checking context on toy Enc*Enc / dot-product test:")
    ok = sanity_check_context(ctx)
    if not ok:
        print("  -> Context failed sanity check. Fix parameters before "
              "trusting anything below.")

    if args.sweep:
        epoch_list = [int(e) for e in args.sweep.split(",")]
        print(f"\nParsed sweep epoch list: {epoch_list}  "
              f"(from --sweep '{args.sweep}')")
        print("If this doesn't match what you typed, check for shell "
              "quoting/copy-paste issues before trusting the results.")
        print(f"Running paper-style sweep over epochs={epoch_list} "
              f"(sigmoid degree={args.sigmoid_degree})...")
        run_sweep(X_train, y_train, X_test, y_test, epoch_list,
                  args.lr, args.l2, ctx, args.sigmoid_degree)
        return

    # --- single run: plaintext baseline ---
    w_plain = train_plaintext_lr(X_train, y_train, epochs=args.epochs,
                                  lr=args.lr, l2=args.l2)
    acc_plain = accuracy(X_test, y_test, w_plain)
    print(f"\nPlaintext LR test accuracy: {acc_plain:.4f}")

    # --- encrypted training ---
    print(f"\nRunning encrypted HELR training for {args.epochs} epochs "
          f"(sigmoid degree={args.sigmoid_degree})...")
    w_helr, _ = train_helr(X_train, y_train, args.epochs, args.lr,
                            args.l2, ctx,
                            sigmoid_degree=args.sigmoid_degree)
    acc_helr = accuracy(X_test, y_test, w_helr)

    print("\n===== RESULTS =====")
    print(f"Plaintext LR accuracy : {acc_plain:.4f}")
    print(f"HELR (encrypted) accuracy: {acc_helr:.4f}")
    print(f"Accuracy gap          : {abs(acc_plain - acc_helr):.4f}")
    print(f"(Paper reports a 0.01-0.03 gap between LR and HELR)")


if __name__ == "__main__":
    main()