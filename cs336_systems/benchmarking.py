"""
Benchmarking script for Transformer model forward and backward passes.

This script performs end-to-end benchmarking of a Transformer model with
configurable hyperparameters and timing options.
"""

import argparse
import timeit
from typing import Literal
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import pandas as pd
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def generate_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random batch of input tokens and target tokens.
    
    Args:
        batch_size: Number of samples in the batch
        context_length: Sequence length
        vocab_size: Size of the vocabulary
        device: Device to create tensors on
    
    Returns:
        Tuple of (input_tokens, target_tokens)
    """
    input_tokens = torch.randint(
        0, vocab_size, (batch_size, context_length), device=device
    )
    target_tokens = torch.randint(
        0, vocab_size, (batch_size, context_length), device=device
    )
    return input_tokens, target_tokens


def run_benchmark(
    model: nn.Module,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    warmup_steps: int,
    num_steps: int,
    mode: Literal["forward", "forward_backward"],
    device: str = "cuda",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """
    Benchmark the model's forward pass or forward+backward pass.
    
    Args:
        model: The model to benchmark
        batch_size: Batch size for random data
        context_length: Sequence length
        vocab_size: Size of vocabulary
        warmup_steps: Number of warmup iterations before timing
        num_steps: Number of timed iterations
        mode: Either "forward" or "forward_backward"
        device: Device to run on
        optimizer: Optimizer for backward pass (required if mode is "forward_backward")
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval() if mode == "forward" else model.train()
    
    # Loss function for backward pass
    criterion = nn.CrossEntropyLoss()
    
    # Warm-up phase
    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        input_tokens, target_tokens = generate_random_batch(
            batch_size, context_length, vocab_size, device
        )
        
        if mode == "forward":
            with torch.no_grad():
                outputs = model(input_tokens)
        else:  # forward_backward
            if optimizer is None:
                raise ValueError("Optimizer required for forward_backward mode")
            
            optimizer.zero_grad()
            outputs = model(input_tokens)
            
            # Compute loss
            loss = criterion(
                outputs.view(-1, vocab_size),
                target_tokens.view(-1)
            )
            loss.backward()
            optimizer.step()
        
        # Synchronize to ensure operations complete
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Timed phase
    print(f"Running {num_steps} timed steps...")
    times = []
    
    for step in range(num_steps):
        input_tokens, target_tokens = generate_random_batch(
            batch_size, context_length, vocab_size, device
        )
        
        # Start timing
        start_time = timeit.default_timer()
        
        if mode == "forward":
            with torch.no_grad():
                outputs = model(input_tokens)
        else:  # forward_backward
            optimizer.zero_grad()
            outputs = model(input_tokens)
            
            # Compute loss
            loss = criterion(
                outputs.view(-1, vocab_size),
                target_tokens.view(-1)
            )
            loss.backward()
            optimizer.step()
        
        # Synchronize before stopping timer
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Stop timing
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if (step + 1) % 10 == 0:
            print(f"  Completed {step + 1}/{num_steps} steps")
    
    # Compute statistics
    import statistics
    results = {
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times),
        "num_steps": num_steps,
    }
    
    return results


def save_benchmark_report(
    config: dict,
    results: dict,
    num_params: int,
    output_dir: str = "output/benchmarks"
) -> None:
    """
    Save benchmark results to a Markdown table.
    
    Args:
        config: Dictionary of model and benchmark configuration
        results: Dictionary of benchmark results
        num_params: Total number of model parameters
        output_dir: Directory to save the report
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for the report
    report_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'vocab_size': config['vocab_size'],
        'context_length': config['context_length'],
        'd_model': config['d_model'],
        'num_layers': config['num_layers'],
        'num_heads': config['num_heads'],
        'd_ff': config['d_ff'],
        'rope_theta': config['rope_theta'],
        'batch_size': config['batch_size'],
        'mode': config['mode'],
        'device': config['device'],
        'total_params_M': num_params / 1e6,
        'mean_time_s': results['mean_time'],
        'median_time_s': results['median_time'],
        'std_time_s': results['std_time'],
        'min_time_s': results['min_time'],
        'max_time_s': results['max_time'],
        'steps_per_sec': 1 / results['mean_time'],
        'tokens_per_step': config['batch_size'] * config['context_length'],
        'tokens_per_sec': (config['batch_size'] * config['context_length']) / results['mean_time'],
    }
    
    # Add GPU memory info if available
    if config['device'] == 'cuda' and torch.cuda.is_available():
        report_data['gpu_allocated_GB'] = torch.cuda.memory_allocated() / 1e9
        report_data['gpu_reserved_GB'] = torch.cuda.memory_reserved() / 1e9
        report_data['gpu_max_allocated_GB'] = torch.cuda.max_memory_allocated() / 1e9
    
    # Load existing results if file exists
    results_file = output_path / 'benchmark_results.csv'
    markdown_file = output_path / 'benchmark_results.md'
    
    if results_file.exists():
        df = pd.read_csv(results_file)
        df = pd.concat([df, pd.DataFrame([report_data])], ignore_index=True)
    else:
        df = pd.DataFrame([report_data])
    
    # Save to CSV
    df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to {results_file}")
    
    # Save to Markdown
    markdown_content = "# Benchmark Results\n\n"
    markdown_content += f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    markdown_content += "## Summary Table\n\n"
    
    # Format the dataframe for better readability
    df_display = df.copy()
    
    # Round numeric columns
    numeric_cols = ['total_params_M', 'mean_time_s', 'median_time_s', 'std_time_s', 
                    'min_time_s', 'max_time_s', 'steps_per_sec', 'tokens_per_sec']
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    if 'gpu_allocated_GB' in df_display.columns:
        df_display['gpu_allocated_GB'] = df_display['gpu_allocated_GB'].round(3)
        df_display['gpu_reserved_GB'] = df_display['gpu_reserved_GB'].round(3)
        df_display['gpu_max_allocated_GB'] = df_display['gpu_max_allocated_GB'].round(3)
    
    markdown_content += df_display.to_markdown(index=False)
    markdown_content += "\n\n## Notes\n\n"
    markdown_content += "- `mean_time_s`: Average time per step in seconds\n"
    markdown_content += "- `tokens_per_sec`: Throughput in tokens per second\n"
    markdown_content += "- `total_params_M`: Total model parameters in millions\n"
    markdown_content += "- `gpu_*_GB`: GPU memory usage in gigabytes\n"
    
    with open(markdown_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"✓ Markdown report saved to {markdown_file}")
    
    # Save individual run details
    run_detail_file = output_path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(run_detail_file, 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'num_params': num_params,
            'report_data': report_data
        }, f, indent=2)
    
    print(f"✓ Detailed run info saved to {run_detail_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer model forward and backward passes"
    )
    
    # Model hyperparameters
    parser.add_argument("--vocab-size", type=int, default=10000,
                        help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=256,
                        help="Maximum context length")
    parser.add_argument("--d-model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=16,
                        help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1344,
                        help="Feed-forward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Benchmarking parameters
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for benchmarking")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Number of warmup steps before timing")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of timed steps")
    parser.add_argument("--mode", type=str, default="forward_backward",
                        choices=["forward", "forward_backward"],
                        help="Benchmark mode: forward only or forward+backward")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run benchmarking on")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate for optimizer (used in forward_backward mode)")
    parser.add_argument("--output-dir", type=str, default="output/benchmarks",
                        help="Directory to save benchmark reports")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to file")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print("=" * 80)
    print("BENCHMARK CONFIGURATION")
    print("=" * 80)
    print("Model Configuration:")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  context_length: {args.context_length}")
    print(f"  d_model: {args.d_model}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  rope_theta: {args.rope_theta}")
    print("\nBenchmark Configuration:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  mode: {args.mode}")
    print(f"  device: {args.device}")
    print("=" * 80)
    
    # Initialize model
    print("\nInitializing model...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model = model.to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Initialize optimizer if needed
    optimizer = None
    if args.mode == "forward_backward":
        print(f"Initializing AdamW optimizer with lr={args.learning_rate}")
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Run benchmark
    print(f"\nStarting benchmark in {args.mode} mode...")
    results = run_benchmark(
        model=model,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        mode=args.mode,
        device=args.device,
        optimizer=optimizer,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Total time: {results['total_time']:.4f} seconds")
    print(f"Number of steps: {results['num_steps']}")
    print("\nPer-step statistics:")
    print(f"  Mean time:   {results['mean_time']:.4f} seconds ({1/results['mean_time']:.2f} steps/sec)")
    print(f"  Median time: {results['median_time']:.4f} seconds")
    print(f"  Std dev:     {results['std_time']:.4f} seconds")
    print(f"  Min time:    {results['min_time']:.4f} seconds")
    print(f"  Max time:    {results['max_time']:.4f} seconds")
    
    # Throughput calculations
    tokens_per_step = args.batch_size * args.context_length
    tokens_per_sec = tokens_per_step / results['mean_time']
    print("\nThroughput:")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Tokens per second: {tokens_per_sec:,.0f}")
    
    # Memory usage (if CUDA)
    if args.device == "cuda":
        print("\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("=" * 80)
    
    # Save benchmark report
    if not args.no_save:
        config_dict = {
            'vocab_size': args.vocab_size,
            'context_length': args.context_length,
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'd_ff': args.d_ff,
            'rope_theta': args.rope_theta,
            'batch_size': args.batch_size,
            'mode': args.mode,
            'device': args.device,
        }
        save_benchmark_report(config_dict, results, num_params, args.output_dir)


if __name__ == "__main__":
    main()
