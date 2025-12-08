# 🏆 N-Queens Problem Solver

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive implementation of the N-Queens problem using multiple algorithms with performance optimizations and visualization.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The N-Queens problem is a classic constraint satisfaction problem where N queens must be placed on an N×N chessboard such that no two queens threaten each other. This project implements multiple algorithms to solve this problem efficiently.

## ✨ Features

- **Multiple Algorithms**: Backtracking, Genetic Algorithm, and Min-Conflicts
- **Performance Optimized**: Pruning, symmetry exploitation, bitboard representations
- **Visualization**: Beautiful chessboard visualizations with Matplotlib
- **Benchmarking**: Compare algorithm performance across different board sizes
- **Modular Design**: Easy to extend with new algorithms
- **Comprehensive Documentation**: Detailed docstrings and type hints

## 🤖 Algorithms

### 1. **Backtracking with Pruning**
- DFS with conflict checking using sets for O(1) operations
- Early pruning when no safe positions remain
- Option to count solutions without storing them

### 2. **Genetic Algorithm**
- Tournament selection for parent selection
- Uniform crossover and mutation operators
- Elitism to preserve best solutions
- Configurable parameters (population size, mutation rate, etc.)

### 3. **Min-Conflicts Heuristic**
- Local search algorithm with random restart
- O(N) conflict checking per move
- Suitable for large board sizes (N > 1000)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Parth-Oza/N-Queen.git
cd N-Queen

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
