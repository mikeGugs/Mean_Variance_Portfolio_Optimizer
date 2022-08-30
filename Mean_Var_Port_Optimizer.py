from tiingo import TiingoClient
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import date, timedelta

client = TiingoClient({'api_key': 'f267669a5603ccc70485209331f588a79359966a'})

class GetUserInput:
    """Gather list of stocks the user wants to include in the optimization,
    and allow user to decide whether they want to include a portfolio dollar
    value to get weights in number of shares instead of percentages."""
    def __init__(self):
        self.list_of_stocks = []
        self.dollar_value = ''

    def get_stocks_list(self):
        """Get the list of stocks."""
        still_entering = True
        while still_entering:
            stock = input("Enter a ticker you'd like to include in your "
                          "portfolio."
                          " Press return to add another one. Enter 'next' to "
                          "move on.\n")
            stock = stock.upper()
            if stock == 'NEXT' or stock == '':
                still_entering = False
            else:
                self.list_of_stocks.append(stock)

    def include_dollar_value(self):
        """Find out if user wants to include dollar value. If yes, get the
        dollar value."""
        dollar_value_question = input("Do you want to include a total "
                                      "dollar value of your portfolio to receive "
                                      "weights in number of shares instead of "
                                      "percentages? (answer 'yes' or 'no') \n")

        if dollar_value_question == 'yes':
            dollar_value = int(input("What is the dollar value of your "
                                     "portfolio?\n"))
            self.dollar_value = dollar_value

class MeanVarConfig:
    """Initialize configurations for gathering and processing data."""
    def __init__(self):
        self.today = date.today()
        self.yesterday = (self.today - timedelta(days=1)).strftime('%Y%m%d')
        self.five_years_ago = (self.today - timedelta(days=(
                365*5))).strftime('%Y%m%d')

class GatherData:
    """Gather and organize all of the data from the user-selected
    list of stocks that will be used for portfolio optimization."""
    def __init__(self, user_input, config):
        self.stocks_history = client.get_dataframe(user_input.list_of_stocks,
                                                   metric_name='adjClose',
                                                   startDate=
                                                   config.five_years_ago,
                                                   endDate=config.yesterday,
                                                   frequency='daily',
                                                   )
        self.statistics = {
                            'stocks': user_input.list_of_stocks,
                            'daily_returns': '',
                            'expected_returns': '',
                            'sigma': '',
                            'cov_matrix': '',
                            'correlations': '',
        }

    def calculate_statistics(self, user_input):
        self.statistics['daily_returns'] = self.stocks_history[
            user_input.list_of_stocks].pct_change()
        self.statistics['expected_returns'] = \
            expected_returns.mean_historical_return(self.stocks_history)
        self.statistics['sigma'] = self.statistics['daily_returns'].std() * (
            252 ** .5) * 100
        self.statistics['cov_matrix'] = risk_models.sample_cov(
            self.stocks_history)
        self.statistics['correlations'] = self.statistics['daily_returns'].corr()
        return self.statistics

class DoOptimization:
    """Run the optimizer, store the efficient frontier in a class attribute
    to be used later for plotting. Show the user portfolio weights and other
    important info."""
    def __init__(self):
        self.efficient_frontier = ''
        self.max_sharpe_weights = ''
        self.min_vol_port = ''

    def max_sharpe_long_only_portfolio(self, statistics):
        efficient_frontier = EfficientFrontier(statistics['expected_returns'],
                                               statistics['cov_matrix'],
                                               weight_bounds=(0, 1),
                                               )
        efficient_frontier.max_sharpe()
        self.max_sharpe_weights = efficient_frontier.clean_weights()
        print('These are the weights of the long-only, maximum Sharpe ratio '
              'portfolio.')
        print(self.max_sharpe_weights)
        efficient_frontier.portfolio_performance(verbose=True)
        return self.max_sharpe_weights

    def min_variance_portfolio(self, statistics):
        print('These are the weights of the long-only, minimum volatility portfolio:')
        efficient_frontier = EfficientFrontier(statistics['expected_returns'],
                                               statistics['cov_matrix'],
                                               weight_bounds=(0, 1),
                                               )
        self.min_vol_port = efficient_frontier.min_volatility()
        print(self.min_vol_port)
        efficient_frontier.portfolio_performance(verbose=True)
        return self.min_vol_port

    def reset_efficient_frontier(self, statistics):
        """Calling max_sharpe and min_volatility on the EF alters the EF.
        This function helps store a clean EF to be used in other parts of the
         program."""
        efficient_frontier = EfficientFrontier(statistics['expected_returns'],
                                               statistics['cov_matrix'],
                                               weight_bounds=(0, 1),
                                               )
        self.efficient_frontier = efficient_frontier


    def show_important_info(self, weights, stocks_history, user_input):
        print("These are the previous days closing prices:")
        closing_prices = get_latest_prices(stocks_history)
        print(closing_prices)
        if user_input:
            discrete = DiscreteAllocation(weights,
                                          closing_prices,
                                          total_portfolio_value=user_input,
                                          short_ratio=None)
            allocation, leftover = discrete.greedy_portfolio()
            print('These are the portfolio weights in shares: \n',
                  allocation)
            print(f"You'll have this much cash leftover as remainder: $"
                  f"{leftover} \n")

class MakePlots:
    @staticmethod
    def plot_efficient_frontier(efficient_frontier, statistics):
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 2, 2)
        plotting.plot_efficient_frontier(efficient_frontier,
                                         ax=ax,
                                         show_assets=True,
                                         )
        # create a graph showing the efficient frontier with 5000 random portfolios
        efficient_frontier.max_sharpe()
        ret_tangent, std_tangent, _ = efficient_frontier.portfolio_performance()
        plt.subplot(1,2,2)
        plt.scatter(std_tangent,
                    ret_tangent,
                    marker='*',
                    s=100,
                    c="r",
                    label="Max Sharpe",
                    )
        # Generate random portfolios and plot them
        n_samples = 5000
        w = np.random.dirichlet(np.ones(len(statistics['expected_returns'])),
                                n_samples)
        returns = w.dot(statistics['expected_returns'])
        std_devs = np.sqrt(np.diag(w @ statistics['cov_matrix'] @ w.T))
        sharpes = returns / std_devs
        ax1 = plt.subplot(1, 2, 2)
        plt.scatter(std_devs, returns, marker=".", c=sharpes, cmap="viridis_r")
        ax1.set_title("Efficient Frontier")
        ax1.legend()
        ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def risk_vs_return_bar_graph(statistics):
        plt.figure(figsize=(10, 5))
        """Creates a bar graph that will show the trade-off of risk vs
        return on each stock in the portfolio."""
        n = 1
        t = 2
        d = len(statistics['stocks'])
        w = .8
        x_values_one = [t * element + w * n for element in range(d)]

        n=2
        x_values_two = [t * element + w * n for element in range(d)]

        ax = plt.subplot(2,2,1)
        plt.bar(x_values_one,
                statistics['expected_returns'],
                color='green',
                )
        plt.bar(x_values_two,
                statistics['sigma'],
                color='red',
                )
        plt.title('Expected Returns vs Volatility')
        plt.xlabel('Stocks')
        plt.ylabel('Expected Return (annualized) vs Volatility (annualized '
                   'sigma')
        plt.legend(['Expected Return', 'Volatility'])
        ax.set_xticks(x_values_one)
        ax.set_xticklables(statistics['stocks'])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.show()


if __name__ == '__main__':
    users_input = GetUserInput()
    users_input.get_stocks_list()
    users_input.include_dollar_value()
    config = MeanVarConfig()
    data = GatherData(users_input, config)
    data.calculate_statistics(users_input)
    optimize = DoOptimization()
    max_sharpe = optimize.max_sharpe_long_only_portfolio(data.statistics)
    show_info = optimize.show_important_info(optimize.max_sharpe_weights,
                                             data.stocks_history,
                                             users_input.dollar_value)
    min_variance = optimize.min_variance_portfolio(data.statistics)
    optimize.show_important_info(optimize.min_vol_port,
                                 data.stocks_history,
                                 users_input.dollar_value)
    plot = MakePlots()
    optimize.reset_efficient_frontier(data.statistics)
    plot.plot_efficient_frontier(optimize.efficient_frontier, data.statistics)
    # MakePlots.risk_vs_return_bar_graph(data.statistics)