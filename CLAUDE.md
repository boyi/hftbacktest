# HftBacktest - Claude Code Context

## Project Overview

**HftBacktest** is a high-frequency trading backtesting and live trading framework written primarily in Rust, with Python bindings. It focuses on providing accurate market replay-based backtesting by accounting for feed latency, order latency, and order queue positions for realistic order fill simulation.

**Status**: Active development, breaking changes may occur. Live bot features are experimental.

**Original Repository**: https://github.com/nkaz001/hftbacktest
**Forked Repository**: https://github.com/boyi/hftbacktest

## Key Capabilities

### Backtesting
- Complete tick-by-tick simulation with customizable time intervals
- Full order book reconstruction (L2 Market-By-Price and L3 Market-By-Order)
- Latency modeling for both feed and order processing
- Order fill simulation considering queue position
- Multi-asset and multi-exchange backtesting
- Data fusion for combining different data streams

### Live Trading
- Deploy live trading bots using the same algorithm code
- Currently supports: Binance Futures, Bybit
- External connectors through IPC for multiple bots via unified connection

### Languages
- **Rust**: Core implementation, high-performance backtesting and live trading
- **Python**: High-level interface via `py-hftbacktest`, works with Numba JIT

## Project Structure

```
hftbacktest/
├── hftbacktest/           # Core Rust library
│   ├── src/
│   │   ├── backtest/     # Backtesting engine
│   │   ├── depth/        # Order book depth management
│   │   ├── live/         # Live trading components
│   │   ├── types.rs      # Core data types
│   │   └── lib.rs        # Library entry point
│   └── examples/         # Rust examples
├── py-hftbacktest/        # Python bindings
│   └── hftbacktest/
│       ├── binding.py    # Rust-Python bindings
│       ├── order.py      # Order management
│       ├── data/         # Data utilities
│       └── stats/        # Performance statistics
├── connector/             # Exchange connectors (IPC-based)
│   └── src/
│       ├── binancefutures/
│       ├── bybit/
│       └── fuse.rs       # Data fusion
├── collector/             # Market data collectors
│   └── src/
│       ├── binance/
│       ├── binancefuturescm/
│       ├── binancefuturesum/
│       └── bybit/
├── examples/              # Jupyter notebook tutorials & examples
└── docs/                  # Documentation
```

## Key Files

### Rust Core
- `hftbacktest/src/lib.rs` - Main library entry point
- `hftbacktest/src/types.rs` - Core type definitions (33KB, extensive)
- `hftbacktest/src/backtest/` - Backtesting engine implementation
- `hftbacktest/src/live/` - Live trading infrastructure

### Python
- `py-hftbacktest/hftbacktest/__init__.py` - Python package entry
- `py-hftbacktest/hftbacktest/binding.py` - Rust bindings (79KB)
- `py-hftbacktest/hftbacktest/order.py` - Order management interface

### Live Trading
- `connector/src/main.rs` - Connector service entry point
- `connector/src/binancefutures/` - Binance Futures connector
- `connector/src/bybit/` - Bybit connector

## Common Workflows

### Backtesting Strategy (Python)
1. Prepare market data (L2/L3 order book + trades)
2. Define strategy using Numba JIT functions
3. Initialize backtester with latency models
4. Run backtest with `hbt.elapse()` loop
5. Analyze performance using built-in stats

### Live Trading (Rust)
1. Implement strategy using same interface as backtest
2. Configure exchange connector
3. Deploy using connector service
4. Monitor via IPC communication

### Data Collection
1. Use `collector` binary to record market data
2. Supports Binance Futures (USD-M, COIN-M), Bybit, Binance Spot
3. Outputs in HftBacktest data format

## Development Areas (from ROADMAP.md)

### High Priority
- [ ] Additional queue position models
- [ ] Order modification feature (partial)
- [ ] Different latencies for various order operations
- [X] Parallel data loading
- [X] Vector-based L2 depth implementation

### Live Trading Expansion
- [ ] Binance Futures Websocket Order APIs (currently REST)
- [ ] Additional exchanges: OKX, Hyperliquid, Coinbase, Kraken
- [ ] Level 3 Market-By-Order support for live

### Python Features
- [ ] Enhanced performance metrics and visualization
- [ ] Live trading support in Python

## Data Format

**Important**: Rust and Python use different data formats.
- See: `docs/tutorials/Data Preparation.ipynb`
- Example data: https://reach.stratosphere.capital/data/usdm/

## Testing & Examples

### Jupyter Notebooks (examples/)
- `Getting Started.ipynb` - Basic introduction
- `High-Frequency Grid Trading.ipynb` - Complete workflow
- `GLFT Market Making Model and Grid Trading.ipynb` - GLFT model
- `Market Making with Alpha - *.ipynb` - Alpha strategies (Basis, APT, Order Book Imbalance)
- `Level-3 Backtesting.ipynb` - Market-By-Order backtesting
- `Queue-Based Market Making in Large Tick Size Assets.ipynb`

### Python Examples
- `examples/example.py` - Basic example
- `examples/example_hyperliquid.py` - Hyperliquid integration
- `examples/example_mexc.py` - MEXC integration

## Build Configuration

### Cargo Workspace
- Members: `hftbacktest`, `hftbacktest-derive`, `py-hftbacktest`, `collector`, `connector`
- Release profile: Aggressive optimizations (LTO, single codegen unit)
- Development profile: Fast compilation (256 codegen units)

## Documentation

- **General docs**: https://hftbacktest.readthedocs.io/
- **Rust API docs**: https://docs.rs/hftbacktest/latest/hftbacktest/
- **Repository**: https://github.com/nkaz001/hftbacktest
- **Python package**: https://pypi.org/project/hftbacktest
- **Rust crate**: https://crates.io/crates/hftbacktest

## Git Workflow

### Remotes
- `origin`: Your fork at https://github.com/boyi/hftbacktest.git
- `upstream`: Original repo at https://github.com/nkaz001/hftbacktest.git

### Typical Commands
```bash
# Push changes to your fork
git push origin master

# Sync from upstream
git pull upstream master

# Create feature branch
git checkout -b feature-name
git push origin feature-name
```

## Architecture Notes

### Latency Modeling
- Feed latency: Time from exchange to strategy
- Order latency: Time from strategy to exchange + response
- Can adjust for different geographic locations

### Order Queue Position
- Critical for realistic fill simulation
- Uses probabilistic models
- Accounts for queue position in limit order book

### Multi-Asset Trading
- Support for correlated asset strategies
- Cross-exchange market making
- Basis trading, statistical arbitrage

## Key Dependencies (from Cargo.lock)
- `tokio`: Async runtime
- `serde`: Serialization
- `tracing`: Logging infrastructure
- `pyo3`: Python bindings (for py-hftbacktest)
- Exchange client libraries (binance-spot-connector-rust, etc.)

## Performance Considerations
- Rust core ensures high performance
- Numba JIT for Python strategies
- Parallel data loading during backtesting
- Vector-based market depth for fast L2 operations
- Zero-copy architecture where possible

## Contributing
- Open issues/discussions on GitHub
- See ROADMAP.md for contribution ideas
- Breaking changes expected during development

## License
MIT License