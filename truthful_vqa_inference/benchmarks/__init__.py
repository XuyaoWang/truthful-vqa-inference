"""
Benchmarks package initialization.

This module initializes the benchmarks package and handles registration of all evaluators.
"""

# First import the registry
from truthful_vqa_inference.benchmarks.registry import BenchmarkRegistry

import truthful_vqa_inference.benchmarks.truthful_vqa
import truthful_vqa_inference.benchmarks.truthful_vqa_ece
import truthful_vqa_inference.benchmarks.truthful_vqa_low
import truthful_vqa_inference.benchmarks.truthful_vqa_medium
import truthful_vqa_inference.benchmarks.truthful_vqa_high
