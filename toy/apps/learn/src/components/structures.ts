export interface Structure {
  key: string;
  name: string;
  idea: string;
  laws: string[];
  enables: string;
}

/** Laws are consequences of structure — pick a structure, see what it buys. */
export class Structures {
  sel = "monoid";

  list: Structure[] = [
    {
      key: "monoid",
      name: "monoid",
      idea: "one associative operation, with an identity",
      laws: ["(X·Y)·Z ≡ X·(Y·Z)", "X·I ≡ X"],
      enables: "re-bracketing a chain — the matrix-chain ordering win",
    },
    {
      key: "comonoid",
      name: "comonoid",
      idea: "a way to duplicate a value: Δ(x) = (x, x)",
      laws: ["⟨f, g⟩ ≡ (f × g) ∘ Δ", "Δ is natural"],
      enables: "pairing — k shared-input projections become one GEMM plus split views",
    },
    {
      key: "trace",
      name: "traced monoid",
      idea: "a fold with feedback: hₜ = A·hₜ₋₁ + xₜ",
      laws: ["scan ≡ tree-reduce", "the fold is associative"],
      enables: "the Blelloch scan lift, streaming softmax",
    },
    {
      key: "product",
      name: "product",
      idea: "⟨f₁,…,f_k⟩ = (×fᵢ) ∘ Δ",
      laws: ["the product law", "naturality of Δ"],
      enables: "fusing sibling branches that share one input",
    },
  ];

  get current(): Structure {
    return this.list.find((s) => s.key === this.sel) ?? this.list[0];
  }

  pick(key: string): void {
    this.sel = key;
  }
}
