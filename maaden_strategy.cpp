// Daily trading algorithm for Maaden (1211.SR) - C++ port of maaden_strategy.py
//
// Same rules as the Python version: SMA10/30 crossover entry (filtered by
// RSI14 < 85), 8% trailing stop exit. This program runs the monthly-DCA
// scenario: 990 SAR contributed on the first trading day of every month,
// starting 2024-01-01, and compares the strategy against simply buying
// shares with every contribution (buy & hold).
//
// Price data is read from data_1211SR.csv (Date,Close), exported from the
// same Yahoo Finance source the Python version uses - C++ has no built-in
// way to fetch that data itself.
//
// Build:  g++ -O2 -std=c++17 -o maaden_strategy maaden_strategy.cpp
// Run:    ./maaden_strategy data_1211SR.csv 990

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

constexpr int FAST_SMA = 10;
constexpr int SLOW_SMA = 30;
constexpr int RSI_PERIOD = 14;
constexpr double RSI_MAX_ENTRY = 85.0;
constexpr double TRAILING_STOP_PCT = 0.08;
constexpr double COMMISSION_PCT = 0.001;

struct Row {
    std::string date;
    double close;
    double sma_fast = std::nan("");
    double sma_slow = std::nan("");
    double rsi = std::nan("");
    bool trend_up = false;
    bool entry_signal = false;
    bool exit_signal = false;
    bool is_contribution = false;
};

struct Trade {
    std::string entry_date, exit_date;
    double entry_price = 0, exit_price = 0;
    double return_pct = 0;
    bool closed = false;
};

std::vector<Row> load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(f, line);  // header
    std::vector<Row> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto comma = line.find(',');
        Row r;
        r.date = line.substr(0, comma);
        r.close = std::stod(line.substr(comma + 1));
        rows.push_back(r);
    }
    return rows;
}

void compute_indicators(std::vector<Row>& rows) {
    int n = static_cast<int>(rows.size());

    // Simple moving averages
    for (int i = 0; i < n; ++i) {
        if (i >= FAST_SMA - 1) {
            double sum = 0;
            for (int k = i - FAST_SMA + 1; k <= i; ++k) sum += rows[k].close;
            rows[i].sma_fast = sum / FAST_SMA;
        }
        if (i >= SLOW_SMA - 1) {
            double sum = 0;
            for (int k = i - SLOW_SMA + 1; k <= i; ++k) sum += rows[k].close;
            rows[i].sma_slow = sum / SLOW_SMA;
        }
    }

    // RSI14, matching pandas .ewm(alpha=1/period, adjust=False).mean() semantics:
    // the recursion starts at the first real price delta (index 1), and the
    // first RSI_PERIOD deltas are needed before a value is reported (min_periods).
    double alpha = 1.0 / RSI_PERIOD;
    double avg_gain = 0, avg_loss = 0;
    int valid_count = 0;
    for (int i = 1; i < n; ++i) {
        double delta = rows[i].close - rows[i - 1].close;
        double gain = std::max(delta, 0.0);
        double loss = std::max(-delta, 0.0);
        if (valid_count == 0) {
            avg_gain = gain;
            avg_loss = loss;
        } else {
            avg_gain = avg_gain * (1 - alpha) + gain * alpha;
            avg_loss = avg_loss * (1 - alpha) + loss * alpha;
        }
        valid_count++;
        if (valid_count >= RSI_PERIOD) {
            double rs = avg_loss == 0 ? std::numeric_limits<double>::infinity() : avg_gain / avg_loss;
            rows[i].rsi = 100.0 - 100.0 / (1.0 + rs);
        }
    }

    // Trend / signals - undefined SMA comparisons are treated as "false",
    // matching how pandas evaluates comparisons against NaN.
    bool prev_trend_up = false;
    for (int i = 0; i < n; ++i) {
        bool has_smas = !std::isnan(rows[i].sma_fast) && !std::isnan(rows[i].sma_slow);
        bool trend_up = has_smas && (rows[i].sma_fast > rows[i].sma_slow);
        bool cross_up = trend_up && !prev_trend_up;
        bool cross_down = !trend_up && prev_trend_up;

        bool rsi_ok = !std::isnan(rows[i].rsi) && rows[i].rsi < RSI_MAX_ENTRY;
        rows[i].entry_signal = cross_up && rsi_ok;
        rows[i].exit_signal = cross_down;
        rows[i].trend_up = trend_up;

        prev_trend_up = trend_up;
    }

    // Mark first trading day of each calendar month as a contribution date.
    std::string prev_month;
    for (auto& r : rows) {
        std::string month = r.date.substr(0, 7);  // "YYYY-MM"
        if (month != prev_month) {
            r.is_contribution = true;
            prev_month = month;
        }
    }
}

struct DcaResult {
    double total_invested = 0;
    double final_value = 0;
    double profit = 0;
    double return_pct = 0;
    int num_trades = 0;
    double win_rate_pct = 0;
    std::vector<Trade> trades;
};

DcaResult dca_backtest(const std::vector<Row>& rows, double monthly_amount) {
    double cash = 0, shares = 0, entry_price = 0, peak_price = 0, total_invested = 0;
    std::vector<Trade> trades;

    for (const auto& row : rows) {
        double price = row.close;

        if (row.is_contribution) {
            cash += monthly_amount;
            total_invested += monthly_amount;
        }

        if (shares > 0) {
            peak_price = std::max(peak_price, price);
            double change = (price - entry_price) / entry_price;
            bool trail_hit = price <= peak_price * (1 - TRAILING_STOP_PCT);

            if (trail_hit || row.exit_signal) {
                cash += shares * price * (1 - COMMISSION_PCT);
                Trade t;
                t.exit_date = row.date;
                t.exit_price = price;
                t.return_pct = change * 100;
                t.closed = true;
                trades.push_back(t);
                shares = 0;
            }
        } else if (row.entry_signal) {
            double new_shares = std::floor((cash * (1 - COMMISSION_PCT)) / price);
            if (new_shares > 0) {
                cash -= new_shares * price * (1 + COMMISSION_PCT);
                shares = new_shares;
                entry_price = price;
                peak_price = price;
            }
        }
    }

    double final_value = cash + shares * rows.back().close;

    int completed = 0, wins = 0;
    for (const auto& t : trades) {
        if (t.closed) {
            completed++;
            if (t.return_pct > 0) wins++;
        }
    }

    DcaResult res;
    res.total_invested = total_invested;
    res.final_value = final_value;
    res.profit = final_value - total_invested;
    res.return_pct = total_invested > 0 ? (final_value / total_invested - 1) * 100 : 0;
    res.num_trades = completed;
    res.win_rate_pct = completed > 0 ? (100.0 * wins / completed) : 0;
    res.trades = trades;
    return res;
}

struct BuyHoldResult {
    double total_invested = 0;
    double final_value = 0;
    double profit = 0;
    double return_pct = 0;
    double shares = 0;
};

BuyHoldResult dca_buy_and_hold(const std::vector<Row>& rows, double monthly_amount) {
    double cash = 0, shares = 0, total_invested = 0;
    for (const auto& row : rows) {
        double price = row.close;
        if (row.is_contribution) {
            cash += monthly_amount;
            total_invested += monthly_amount;
            double new_shares = std::floor(cash / price);
            if (new_shares > 0) {
                cash -= new_shares * price;
                shares += new_shares;
            }
        }
    }
    double final_value = shares * rows.back().close + cash;
    BuyHoldResult res;
    res.total_invested = total_invested;
    res.final_value = final_value;
    res.profit = final_value - total_invested;
    res.return_pct = total_invested > 0 ? (final_value / total_invested - 1) * 100 : 0;
    res.shares = shares;
    return res;
}

int main(int argc, char** argv) {
    std::string csv_path = argc > 1 ? argv[1] : "data_1211SR.csv";
    double monthly_amount = argc > 2 ? std::stod(argv[2]) : 990.0;

    std::vector<Row> rows = load_csv(csv_path);
    compute_indicators(rows);

    int months = 0;
    for (const auto& r : rows) if (r.is_contribution) months++;

    DcaResult strat = dca_backtest(rows, monthly_amount);
    BuyHoldResult bh = dca_buy_and_hold(rows, monthly_amount);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Monthly DCA of " << monthly_amount << " SAR into Maaden (1211.SR) [C++] ===\n";
    std::cout << "Period: " << rows.front().date << " to " << rows.back().date
              << " (" << months << " contributions)\n";
    std::cout << "Total invested: " << strat.total_invested << " SAR\n\n";

    std::cout << "--- Strategy (SMA crossover + RSI + trailing stop) ---\n";
    std::cout << "Final value: " << strat.final_value << " SAR\n";
    std::cout << "Profit:      " << strat.profit << " SAR (" << strat.return_pct << "%)\n";
    std::cout << "Trades: " << strat.num_trades << ", Win rate: " << strat.win_rate_pct << "%\n\n";

    std::cout << "--- Buy & hold (invest every contribution immediately) ---\n";
    std::cout << "Final value: " << bh.final_value << " SAR\n";
    std::cout << "Profit:      " << bh.profit << " SAR (" << bh.return_pct << "%)\n";
    std::cout << "Shares held: " << static_cast<long long>(bh.shares) << "\n";

    return 0;
}
