"""Patch rejection_sampler.py inside the container to fix Eagle3 PTSS overflow.

Root cause: sample_recovered_tokens_kernel loads tl.arange(0, PADDED_VOCAB_SIZE)
where PADDED_VOCAB_SIZE = next_power_of_2(vocab_size) = 262144 for gpt-oss-20b's
~200K vocab. This creates ~4 arrays × 262144 × 4 bytes ≈ 4MB per-thread scratch
space (PTSS), exceeding the Intel XPU hardware limit of 256KB.
Error: ZE_RESULT_ERROR_MODULE_BUILD_FAILURE in sample_recovered_tokens_kernel.

Fix: Replace the single-shot vocab load+argmax with a chunked argmax over
BLOCK_VOCAB=4096-element tiles. Each tile uses ~64KB scratch (well under 256KB).
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py"


def apply():
    with open(PATH, "r") as f:
        content = f.read()

    changed = False

    # ------------------------------------------------------------------
    # Patch 1: call site — replace triton.next_power_of_2(vocab_size)
    # with 4096 (BLOCK_VOCAB tile size)
    # ------------------------------------------------------------------
    OLD1 = "        triton.next_power_of_2(vocab_size),\n        NO_DRAFT_PROBS=draft_probs is None,"
    NEW1 = "        4096,  # BLOCK_VOCAB: tile size for Intel XPU PTSS fix (was next_power_of_2)\n        NO_DRAFT_PROBS=draft_probs is None,"
    if OLD1 in content:
        content = content.replace(OLD1, NEW1, 1)
        print("Patch 1 (call site) applied")
        changed = True
    elif NEW1 in content:
        print("Patch 1 (call site) already applied, skipping")
    else:
        print("ERROR: Patch 1 — call site pattern not found")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Patch 2: kernel signature — rename PADDED_VOCAB_SIZE -> BLOCK_VOCAB
    # ------------------------------------------------------------------
    OLD2 = "    PADDED_VOCAB_SIZE: tl.constexpr,\n    NO_DRAFT_PROBS: tl.constexpr,\n):"
    NEW2 = "    BLOCK_VOCAB: tl.constexpr,  # tile size; was PADDED_VOCAB_SIZE\n    NO_DRAFT_PROBS: tl.constexpr,\n):"
    if OLD2 in content:
        content = content.replace(OLD2, NEW2, 1)
        print("Patch 2 (signature) applied")
        changed = True
    elif NEW2 in content:
        print("Patch 2 (signature) already applied, skipping")
    else:
        print("ERROR: Patch 2 — kernel signature not found")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Patch 3: kernel body — chunked argmax replacing monolithic load
    # We do a line-number based replacement: find the line that sets
    # vocab_offset = tl.arange(0, ...) and replace from there to end-of-func.
    # ------------------------------------------------------------------
    MARKER = "    vocab_offset = tl.arange(0, "
    if MARKER in content:
        start = content.index(MARKER)
        # Find end of function = last tl.store line in this function
        end_marker = "    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)\n"
        assert end_marker in content, "tl.store end marker not found"
        end = content.index(end_marker) + len(end_marker)
        old_body = content[start:end]

        new_body = (
            "    # NOTE(Intel XPU): Original code loaded tl.arange(0, next_power_of_2(vocab_size))\n"
            "    # = 262144 elements for gpt-oss-20b ~200K vocab.\n"
            "    # ~4 arrays × 262144 × 4 bytes ≈ 4MB PTSS per work-item, exceeding the\n"
            "    # Intel XPU 256KB limit → ZE_RESULT_ERROR_MODULE_BUILD_FAILURE.\n"
            "    # Fix: chunked argmax over BLOCK_VOCAB-sized tiles (~64KB per tile).\n"
            "    if NO_DRAFT_PROBS:\n"
            "        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)\n"
            "\n"
            "    best_val = float(\"-inf\")\n"
            "    best_idx = 0\n"
            "    for i in range(tl.cdiv(vocab_size, BLOCK_VOCAB)):\n"
            "        chunk_start = i * BLOCK_VOCAB\n"
            "        vocab_offset = chunk_start + tl.arange(0, BLOCK_VOCAB)\n"
            "        mask = vocab_offset < vocab_size\n"
            "        if NO_DRAFT_PROBS:\n"
            "            prob = tl.load(\n"
            "                target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,\n"
            "                mask=(mask & (vocab_offset != draft_token_id)),\n"
            "                other=0.0,\n"
            "            )\n"
            "        else:\n"
            "            draft_prob = tl.load(\n"
            "                draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,\n"
            "                mask=mask,\n"
            "                other=0.0,\n"
            "            )\n"
            "            target_prob = tl.load(\n"
            "                target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,\n"
            "                mask=mask,\n"
            "                other=0.0,\n"
            "            )\n"
            "            prob = tl.maximum(target_prob - draft_prob, 0.0)\n"
            "        q = tl.load(\n"
            "            q_ptr + req_idx * vocab_size + vocab_offset,\n"
            "            mask=mask,\n"
            "            other=1.0,  # avoid div-by-zero; val forced to -inf for masked\n"
            "        )\n"
            "        val = tl.where(mask, prob / q, float(\"-inf\"))\n"
            "        chunk_max_val = tl.max(val, axis=0)\n"
            "        chunk_max_idx = tl.argmax(val, axis=0).to(tl.int32) + chunk_start\n"
            "        if chunk_max_val > best_val:\n"
            "            best_val = chunk_max_val\n"
            "            best_idx = chunk_max_idx\n"
            "\n"
            "    tl.store(output_token_ids_ptr + start_idx + pos, best_idx)\n"
        )
        content = content[:start] + new_body + content[end:]
        print("Patch 3 (kernel body) applied")
        changed = True
    else:
        if "chunked argmax" in content:
            print("Patch 3 (kernel body) already applied, skipping")
        else:
            print("ERROR: Patch 3 — MARKER not found")
            sys.exit(1)

    if changed:
        with open(PATH, "w") as f:
            f.write(content)
        print("File written successfully")
    else:
        print("No changes needed (all patches already applied)")


if __name__ == "__main__":
    apply()
