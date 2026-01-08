#!/usr/bin/env python3
import sys
sys.path.insert(0, '/bigtemp/nzj6jt/workspace/project/mdlm_llm_search')

from properties.quality_property import count_quality_violations, calculate_mean_word_length, calculate_symbol_to_word_ratio, calculate_alphabetic_word_ratio
from properties.repetition_property import count_repetition_violations
from properties.word_repetition_property import count_word_repetition_violations, calculate_word_ngram_repetition_ratio

text = """to#"#"#, get and run away from#"#",'-""#, will handcuff you, and get and run from#"#"#!"!!'!!!!"!!'!'!!!'!!'!"!"!"!!'!'!!!'!!!!!!'!!!!!'!!!!!!!'!!"""

print("="*80)
print("Text:", text)
print("="*80)
print()

# Quality
print("QUALITY METRICS:")
mean_len = calculate_mean_word_length(text)
symbol_ratio = calculate_symbol_to_word_ratio(text)
alpha_ratio = calculate_alphabetic_word_ratio(text)
quality_viol = count_quality_violations(text)
print(f"  Mean word length: {mean_len:.2f} (valid: 3-10)")
print(f"  Symbol-to-word ratio: {symbol_ratio:.3f} (valid: <0.1)")
print(f"  Alphabetic ratio: {alpha_ratio:.3f} (valid: >=0.8)")
print(f"  → Quality violations: {quality_viol}")
print()

# Repetition
print("CHARACTER REPETITION:")
repetition_viol = count_repetition_violations(text)
print(f"  → Repetition violations: {repetition_viol}")
print()

# Word repetition
print("WORD REPETITION:")
ratio_3 = calculate_word_ngram_repetition_ratio(text, 3)
ratio_4 = calculate_word_ngram_repetition_ratio(text, 4)
ratio_5 = calculate_word_ngram_repetition_ratio(text, 5)
word_rep_viol = count_word_repetition_violations(text)
print(f"  3-gram ratio: {ratio_3:.3f} (threshold: 0.30)")
print(f"  4-gram ratio: {ratio_4:.3f} (threshold: 0.25)")
print(f"  5-gram ratio: {ratio_5:.3f} (threshold: 0.20)")
print(f"  → Word repetition violations: {word_rep_viol}")
print()

print("="*80)
total_viol = quality_viol + repetition_viol + word_rep_viol
if total_viol > 0:
    print(f"❌ VI PHẠM {total_viol} constraints (yêu cầu ≤0)!")
    print("   → Có BUG trong filtering logic!")
else:
    print("✅ Thỏa mãn tất cả constraints")
print("="*80)
