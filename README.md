# ParLatent: Parallel Latent Space Reasoning

## Overview
This project showcases a small transformer-based model trained to solve sudoku puzzles with parallel latent space reasoning via layer looping. Unlike traditional chain of thought techniques, all reasoning is implicit in latent space with the reasoning tokens being the arbitrary placeholder HOLE. Since the reasoning tokens are static and arbitrary, they do not need to be sampled one at a time by the model and instead can be prefilled simulateously, enabling massively parallel reasoning. I show that the 4 layer model looped 4 times outperforms the 16 layer non-looped variant on sudoku puzzles. Additionally, I ablated 8 layer 8 loops vs non-looped 64 layers and the looped one was demonstrably superior to the nonlooped variant in this case as well.

## Results

![Held-out accuracy by hole count](plots/acc_u4x4.png)

The 4 layer 4 looped variant learns faster and exceeds or matches the 16 layer non-looped at every tested puzzle difficulty.

![Training loss](plots/loss_u4x4.png)

The 4 layer 4 looped variant's average per loop loss quickly falls and ends below the 16 layer non-looped's loss at 0.281 vs 0.300.

![Held-out accuracy by hole count](plots/acc_u8x8.png)

The 8 layer 8 looped variant crushes the 64 layer non-looped one. At 55 holes, the looped ends with a solve accuracy of 66% vs the non-loop's 2%.

![Training loss](plots/loss_u8x8.png)

The 8 layer 8 looped ablation loss graph tells a similar story to the 4 layer 4 looped one with looped's 0.165 vs non-looped's 0.186.

## Architecture

The backbone of the model is multi head attention with rotary position embeddings. The feedforwards each comprise two matmuls with a relu^2 activation instead of the common SwiGLU layer. After each loop, the hidden state is normed and the token embedding is added to preserve the token signal throughout all loops. Cross entropy loss is computed after each loop with the average over all loops as the final loss. This helps propagate the gradient signal throughout all loops as otherwise the rmsnorm on the hidden state each loop can potentially bottleneck the gradient flow.

Below is pseudo code
```
    tok_emb = embeddings[toks]
	x = tok_emb
	
	for l in nloops:
		x = hidden_norm(x) + tok_emb_norm(tok_emb)
		for block in blocks:
			x = block(x)
		logits = proj(proj_norm(x))
		loss += cross_entropy(logits, targets)
	loss /= nloops
```

## Data

The token set is {`<holed>`, `</holed>`, 1-9, _, |, `<solved>`, `</solved>`}.
`<holed>` designates the start of a sudoku puzzle with holes and `</holed>` the end.
_ is the hole token and | is the row separator.
`<solved>` and `</solved>` designate the final solved puzzle.

Input training sequences are formatted like so:
```
[<holed>, 7, _, 1, 6, 4, 3, _, 2, 8, |,
 1, ..., </holed>, <solved>, _, _, _, ..., |, 
 ..., </solved>, <solved>, ..., </solved>]
```

The sequences are generate synthetically with the number of holes in each puzzle is sampled uniformly ~[0, 50]. Each generated puzzle's solution is guaranteed unique.

Output sequences are similar but with all holes of the `<solved>` puzzles filled in with the correct numbers. The model is trained to predict the next token of the output sequences for every input token.

## Training
The muon optimizer is used for the weight matrices while AdamW is used for the embeddings and all other parameters.

The number of loops is chosen to equal the number of `<solved>` puzzles in the training sequences so that each `<solved>` prediction can be a refinement of the previous full `<solved>` prediction. This project uses 4 `<solved>` puzzles and 4 loops for the looped variant and an 8 run as well. The non-looped variant also uses 4 `<solved>`, but with 16 distinct layers.

The learning rate grid {0.005, 0.01, 0.02, 0.04, 0.08} was run for the looped and non-looped architectures over 5000 steps and the best lr was 0.01 for both. The looped variant demonstrated overall faster learning and solution accuracy over the non-looped variant despite having ~4x fewer parameters.