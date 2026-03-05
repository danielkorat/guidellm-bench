"""Fix script: replace the old throughput data block with the new c16 LC block."""
import sys

DASH = '/root/dkorat/guidellm-bench/guidellm_bench/dashboard.py'
SNIPPET = '/root/dkorat/guidellm-bench/c16_snippet.py.txt'

with open(DASH, 'r', encoding='utf-8') as f:
    content = f.read()

START = (
    '    ts = _run_timestamp(out_dir)\n'
    '    conclusions_html = _generate_conclusions(lc_data, throughput_data=throughput_data)'
)
END_SUFFIX = (
    '        throughput_tab_nav = ""\n'
    '        throughput_tab_html = ""\n'
    '        throughput_js = ""\n'
    '\n'
    '    # ------------------------------------------------------------------\n'
    '    # 8k snapshot bars and % delta vs baseline'
)

idx_start = content.find(START)
idx_end   = content.find(END_SUFFIX)
if idx_start < 0:
    print(f"ERROR: START marker not found"); sys.exit(1)
if idx_end < 0:
    print(f"ERROR: END_SUFFIX marker not found"); sys.exit(1)

idx_end += len(END_SUFFIX)
old_block = content[idx_start:idx_end]
print(f"Found old block: [{idx_start}:{idx_end}] ({len(old_block)} chars)")

with open(SNIPPET, 'r', encoding='utf-8') as f:
    new_block = f.read().lstrip('\n')

print(f"New block: {len(new_block)} chars")
print(f"  starts: {repr(new_block[:70])}")
print(f"  ends:   {repr(new_block[-70:])}")

content = content[:idx_start] + new_block + content[idx_end:]

with open(DASH, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nSUCCESS")
