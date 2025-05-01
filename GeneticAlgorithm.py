import time
import matplotlib.pyplot as plt
from numpy.random import randint, rand
import pandas as pd
import numpy as np
import warnings
import logging


logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GeneticAlgorithm:
    def __init__(
        self,
        file_name=None,
        maximize=True,
        population_size=100,
        tournament_size=4,
        crossover_prob=0.5,
        mutation_prob=0.02,
        num_mutations=1,
        elitism=False,
        max_time=60,
        max_gen=2000000000,
        initial_capital=10000,
        operational_cost=1,
    ):
        self.file_name = file_name
        self.maximize = maximize
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_mutations = num_mutations
        self.elitism = elitism
        self.max_time = max_time
        self.max_gen = max_gen
        self.january = (
            pd.read_csv(self.file_name) if self.file_name else self.generate_dataset()
        )
        self.initial_capital = initial_capital
        self.operational_cost = operational_cost
        self.population = list()
        self.historic_data = pd.read_csv("historic_prices.csv")
        self.covariance_matrix = pd.read_csv("historic_covariance.csv")
        self.expected_returns = self.january.mean(axis=0).to_numpy()

    @staticmethod
    def generate_dataset(
        num_stocks=500, start="2024-01-01", end="2024-01-31", freq="D"
    ):
        # Create a date range for all days in January
        date_range = pd.date_range(start=start, end=end, freq=freq)
        data = []
        for date in date_range:
            row = [date] + list(np.random.randint(100, 501, size=num_stocks))
            data.append(row)
        columns = ["Date"] + [str(i) for i in range(1, num_stocks + 1)]
        df = pd.DataFrame(data, columns=columns)
        return df

    def generate_chromosome_randomized_with_restrictions(self, n_assets=500, n_days=14):
        valid = False
        while not valid:
            chromosome = np.zeros(
                (n_assets, n_days), dtype=int
            )  # Initialize chromosome with 0
            total_capital = self.initial_capital  # Initial capital
            stocks_held = np.zeros(n_assets, dtype=int)  # Stocks held for each asset

            for day in range(n_days):
                # Generate a random order of assets to process
                random_asset_order = np.random.permutation(n_assets)

                for asset in random_asset_order:
                    # Current asset price on the given day
                    asset_price = self.january.iloc[day, asset]

                    # Probability of each action (hold, buy, sell)
                    action_prob = np.random.rand()

                    # Hold action (33% probability)
                    if action_prob < 1 / 3:
                        continue  # Do nothing
                    # Buy action (33% probability)
                    elif action_prob < 2 / 3:
                        if (
                            total_capital >= asset_price
                        ):  # Only buy if we have enough capital
                            max_buy = total_capital // asset_price  # Maximum we can buy
                            if asset_price == 0:
                                max_buy = 1
                            else:
                                max_buy = total_capital // asset_price

                            quantity = np.random.randint(
                                1, max_buy + 1
                            )  # Random buy quantity
                            chromosome[asset, day] = quantity  # Record buy action
                            stocks_held[asset] += quantity  # Update held stocks
                            total_capital -= quantity * asset_price  # Deduct capital
                    # Sell action (33% probability)
                    else:
                        if stocks_held[asset] > 0:  # Only sell if we own stocks
                            max_sell = stocks_held[
                                asset
                            ]  # Cannot sell more than we own
                            quantity = np.random.randint(
                                1, max_sell + 1
                            )  # Random sell quantity
                            chromosome[asset, day] = -quantity  # Record sell action
                            stocks_held[asset] -= quantity  # Update held stocks
                            total_capital += (
                                quantity * asset_price
                            )  # Add capital from sale

            # Ensure all remaining stocks are sold on the last day
            for asset in range(n_assets):
                if stocks_held[asset] > 0:  # If there are remaining stocks
                    last_day_price = self.january.iloc[
                        n_days - 1, asset
                    ]  # Price on the last day

                    chromosome[asset, n_days - 1] -= stocks_held[
                        asset
                    ]  # Sell all remaining stocks
                    # print(f"chromosome[asset, n_days - 1] : {chromosome[asset, n_days - 1]} = -stocks_held[asset] : {-stocks_held[asset]} ")
                    total_capital += (
                        stocks_held[asset] * last_day_price
                    )  # Update total capital
                    stocks_held[asset] = 0  # Clear remaining stocks

            # Validate the chromosome
            valid = self.check_chromosome(chromosome)  # Validate chromosome

        return chromosome



    @staticmethod
    def aux_check_chromosome_has_previous_stocks(chromosome):
        for stock in range(chromosome.shape[0]):
            value = 0
            for day in range(
                chromosome.shape[1] - 1
            ):  # -1 because we do not want to check the last day!
                value += chromosome[stock][day]
                if value < 0:
                    # print(stock,day)
                    return False
        return True

    def check_chromosome(self, chromosome):
        total = 10000
        day_sold = 0
        if np.any(chromosome[:, 0] < 0):  # Can´t sell in the first day
            #print("Can´t sell in the first day ")
            return False
        if not GeneticAlgorithm.aux_check_chromosome_has_previous_stocks(
            chromosome
        ):  # before selling stocks we must own them
            #print("Before selling stocks we must own them")
            return False

        for active_index in range(chromosome.shape[1]):  # Iterate through columns(days)
            day_sold, day_spent = 0, 0

            if np.any(chromosome[:, active_index] < 1):  # If it decides to sell
                negative_indexes = np.where(chromosome[:, active_index] < 0)[0]
                for index in negative_indexes:
                    day_sold += (
                        chromosome[index][active_index]
                        * self.january.iloc[active_index, index]
                    )  # Date column is taken into account so +1
            total_before = total
            total -= day_spent + day_sold

            if np.any(chromosome[:, active_index] > 1):  # If it decides to buy
                positive_indexes = np.where(chromosome[:, active_index] > 0)[0]
                for index in positive_indexes:
                    day_spent += (
                        chromosome[index][active_index]
                        * self.january.iloc[active_index, index]
                    )  # Date column is taken into account so +1

                if (
                    day_spent > total
                ):  # Cannot spend more than you have. We first buy, then we can sell.
                    #print(f"Cannot spend more than you have.")
                    return False

            if total < 0:
                #print(f"total<0 : {total} = {total_before} - {day_spent} + {day_sold}")
                return False
            # print(f"Day spent : {day_spent} > total : {total}")
            # print(f"TOTAL: {total},DAY_SOLD: {abs(day_sold)}, DAY_SPENT : {-day_spent}")
        return True

    def get_capital(self, chromosome, asset_index, day_index):
        """
        Calcula el capital restante y las acciones poseídas hasta un día y activo específicos.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - january: DataFrame, precios de los activos por día.
        - asset_index: int, índice del activo hasta el cual se evalúa.
        - day_index: int, índice del día hasta el cual se evalúa.
        - initial_capital: int, capital inicial disponible (por defecto, 10000).

        Returns:
        - capital: float, capital restante hasta el día y activo dados.
        - stocks_held: np.ndarray, acciones poseídas hasta el día y activo dados.
        """
        # Inicializar capital y stocks poseídos
        capital = self.initial_capital
        n_assets = chromosome.shape[0]
        stocks_held = np.zeros(n_assets, dtype=int)

        # Evaluar las decisiones hasta el día y activo seleccionados
        for d in range(day_index + 1):  # Recorre días hasta el día dado
            for a in range(asset_index + 1):  # Recorre activos hasta el índice dado
                if chromosome[a, d] > 0:  # Compra
                    cost = chromosome[a, d] * self.january.iloc[d, a]
                    capital -= cost
                    stocks_held[a] += chromosome[a, d]
                elif chromosome[a, d] < 0:  # Venta
                    revenue = -chromosome[a, d] * self.january.iloc[d, a]
                    capital += revenue
                    stocks_held[a] += chromosome[a, d]  # Resta la venta

        return capital, stocks_held

    def filter_and_choose_random(self, last_day, asset, max_quantity):
        """
        Filtra el DataFrame `january` para seleccionar valores mayores a un día y activo dados,
        y con precios menores que una cantidad específica. Finalmente, escoge uno al azar.

        Parameters:
        - january: pd.DataFrame, precios de los activos (filas: días, columnas: activos).
        - last_day: int, índice del último día considerado.
        - asset: int, índice del activo.
        - max_quantity: float, cantidad máxima para filtrar.

        Returns:
        - (selected_day, selected_asset, price): Tupla con el día, activo y precio seleccionado.
        Devuelve (None, None, None) si no hay resultados que cumplan las condiciones.
        """
        # Filtrar el DataFrame para días y activos posteriores al dado
        filtered_df = self.january.iloc[last_day:-1, asset]

        # Filtrar valores por la cantidad máxima
        filtered_df = filtered_df[filtered_df <= max_quantity]

        # Verificar si hay elementos disponibles después del filtrado
        if filtered_df.empty:
            return None, None, None

        # Escoger un índice aleatorio entre los valores que cumplen las condiciones
        selected_index = np.random.choice(filtered_df.index)

        # Devolver el día, el activo y el precio
        return selected_index, asset, self.january.iloc[selected_index, asset]

    @staticmethod
    def find_lasts_movements(chromosome, day_index, asset_index):
        """
        Encuentra la última y próxima compra y venta realizada en cualquier activo antes de un día específico.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - day_index: int, índice del día.
        - asset_index int, índice del stock.

        Returns:
        -
        """
        results = list()
        asset_found = False
        day_found = False
        # Recorrer días y activos en orden inverso
        for asset in range(asset_index - 1, -1, -1):
            for day in range(day_index - 1, -1, -1):
                if chromosome[asset, day] > 0:  # Si se encuentra una compra
                    last_buy = [asset, day, chromosome[asset, day]]
                    results.append(last_buy)
                    asset_found = True
                    break
            if not asset_found:
                return []
            else:
                break

        # Recorrer días y activos en orden lógico
        for day_ind in range(day + 1, chromosome.shape[1], 1):
            if chromosome[asset, day_ind] < 0:  # Si se encuentra una venta
                results.append([asset, day_ind, chromosome[asset, day_ind]])
                day_found = True
                break

            if not day_found:
                return []

        return results

    @staticmethod
    def generate_mutation_indices(chromosome, num_mutations):
        """
        Genera índices de mutación asegurando que siempre cumplan con los criterios
        de encontrar una última compra válida y una venta válida.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - num_mutations: int, número de mutaciones a generar.

        Returns:
        - list de tuplas (asset_index, day_index) con índices válidos.
        """
        n_assets, n_days = chromosome.shape
        selected_indices = []

        while len(selected_indices) < num_mutations:
            # Generar índices aleatorios
            asset_index = np.random.randint(0, n_assets)
            day_index = np.random.randint(0, n_days)

            try:
                # Comprobar si el índice cumple con los criterios
                movements = GeneticAlgorithm.find_lasts_movements(
                    chromosome, day_index, asset_index
                )
                # Validar si encontró tanto una compra como una venta
                last_buy = any(m[2] > 0 for m in movements)
                next_sale = any(m[2] < 0 for m in movements)

                if last_buy and next_sale:
                    selected_indices.append((asset_index, day_index))
            except:
                continue

        return selected_indices

    def mutate_selected_points(self, chromosome):
        """
        Realiza mutaciones en puntos seleccionados de un cromosoma respetando las restricciones.

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (compra/venta/holdear).
        - january: DataFrame, precios de los activos por día.
        - initial_capital: int, capital inicial disponible (por defecto, 10000).

        Returns:
        - mutated_chromosome: np.ndarray, cromosoma mutado.
        """
        mutated_chromosome = chromosome.copy()
        if self.mutation_prob > np.random.random():
            n_assets, n_days = chromosome.shape
            selected_indices = GeneticAlgorithm.generate_mutation_indices(
                chromosome, num_mutations=self.num_mutations
            )  # [(np.random.randint(0, n_assets), np.random.randint(0, n_days)) for _ in range(num_mutations)]

            for asset_index, day_index in selected_indices:
                # Obtener el capital y las acciones disponibles hasta este momento
                capital, stocks_held = self.get_capital(
                    mutated_chromosome, asset_index, day_index
                )

                # Precio del activo en el día actual
                asset_price = self.january.iloc[day_index, asset_index]

                movements = GeneticAlgorithm.find_lasts_movements(
                    chromosome, day_index, asset_index
                )
                last_asset, last_day, quantity = movements[0]  # Buys
                substract = np.random.randint(1, quantity + 1)
                mutated_chromosome[last_asset][last_day] = (
                    quantity - substract
                )  # Now we must make sure we sell "substract amount" fewer participations of that stock!

                last_sell, last_sell_day, sell_quantity = movements[1]  # Sells
                mutated_chromosome[last_sell][last_sell_day] = (
                    sell_quantity + substract
                )  # now we should have substract*january.iloc[last_day, last_asset] USD more

                extra_usd_earned = substract * self.january.iloc[last_day, last_asset]

                selected_day, selected_asset, price = self.filter_and_choose_random(
                    last_day, last_asset, extra_usd_earned
                )
                max_buy = extra_usd_earned // price
                actives_bought = np.random.randint(
                    1, max_buy + 1
                )  # we should have some USD margin. We must now make sure to sell them before january ends.
                mutated_chromosome[selected_asset][selected_day] = actives_bought

                day_to_sell = np.random.choice(
                    [day for day in range(selected_day, n_days, 1)]
                )  # We select a random day to sell, but not the last one.
                mutated_chromosome[selected_asset][day_to_sell] -= actives_bought

        return mutated_chromosome

    def calculate_weights_from_chromosome(self, cromosoma):
        """
        Calculate portfolio weights based on the chromosome (buy/sell quantities).

        Parameters:
        - cromosoma: np.array, a matrix of buy/sell decisions (positive for buy, negative for sell).

        Returns:
        - weights: np.array, portfolio weights (proportions of capital allocated to each asset).
        """
        # Initialize the net flows (purchases/sales) for each asset
        net_flows = np.zeros(cromosoma.shape[0])  # One entry for each asset

        # Calculate net flows for each asset based on the chromosome
        for asset in range(cromosoma.shape[0]):
            for day in range(cromosoma.shape[1]):
                if cromosoma[asset, day] > 0:  # Buy action
                    net_flows[asset] -= (
                        cromosoma[asset, day] * self.january.iloc[day, asset]
                    )  # Outflow (buy)
                elif cromosoma[asset, day] < 0:  # Sell action
                    net_flows[asset] += (
                        abs(cromosoma[asset, day]) * self.january.iloc[day, asset]
                    )  # Inflow (sell)

        # Normalize to get portfolio weights (capital allocation for each asset)
        total_flows = np.sum(net_flows)
        weights = net_flows / total_flows

        return weights

    def calculate_portfolio_return(self, weights):
        """
        Calculate the expected return of the portfolio.

        Parameters:
        - weights: np.array, portfolio weights (proportions of capital allocated to each asset).

        Returns:
        - portfolio_return: float, expected return of the portfolio.
        """
        return np.dot(weights, self.expected_returns)

    def calculate_portfolio_risk(self, weights):
        """
        Calculate the portfolio risk (variance) using the covariance matrix of returns.

        Parameters:
        - weights: np.array, portfolio weights (proportions of capital allocated to each asset).
        - returns_df: pd.DataFrame, DataFrame of historical asset returns over time.

        Returns:
        - risk: float, portfolio variance.
        """
        # covariance_matrix = returns_df.cov()  # Covariance matrix of historical asset returns
        covariance_matrix = self.covariance_matrix
        risk = np.dot(
            weights.T, np.dot(covariance_matrix, weights)
        )  # Portfolio variance
        return risk

    def CARA(self, chromosome, gamma):
        """
        Calculate the Certainty-Equivalent Risk Aversion (CARA) objective function for a portfolio.

        Parameters:
        - chromosome: np.array, a matrix representing the buy/sell decisions for each asset and day.
        - gamma: float, risk aversion coefficient.

        Returns:
        - cara_value: float, the CARA objective value for the portfolio.
        """
        # 1. Calculate portfolio weights from chromosome (buy/sell quantities)
        weights = self.calculate_weights_from_chromosome(chromosome)

        # 2. Calculate portfolio return (E[R_p])
        portfolio_return = self.calculate_portfolio_return(
            weights
        )  # expected_returns(500,), average daily predictions. expected_returns = january.mean(axis=0).to_numpy()  # Vector de tamaño (500,)

        # 3. Calculate portfolio risk (variance)
        portfolio_risk = self.calculate_portfolio_risk(weights)

        # 4. Calculate CARA objective function
        cara_value = portfolio_return - (gamma / 2) * portfolio_risk

        return cara_value

    def is_better_than(self, a, b):
        return a > b if self.maximize else a < b

    def selection(self, population, fitness):
        chosen = randint(len(population))
        for i in randint(0, len(population), self.tournament_size - 1):
            if self.is_better_than(fitness[i], fitness[chosen]):
                chosen = i
        return population[chosen]

    def get_most_active_stocks(self, chromosome):
        """
        Identifica los dos stocks en los que más dinero se mueve (compras y ventas combinadas).

        Parameters:
        - chromosome: np.ndarray, matriz de decisiones (n_assets x n_days).

        Returns:
        - most_active_stock_1: int, índice del stock con mayor movimiento de dinero.
        - most_active_stock_2: int, índice del stock con segundo mayor movimiento de dinero.
        - money_moved_1: float, cantidad total de dinero movido por el primer stock.
        - money_moved_2: float, cantidad total de dinero movido por el segundo stock.
        """
        # Movimiento de dinero por activo
        money_moved_per_asset = np.sum(np.abs(chromosome * self.january.T), axis=1)

        # Obtener los índices de los dos stocks con mayor movimiento
        sorted_indices = np.argsort(money_moved_per_asset)[
            ::-1
        ]  # Ordenar en orden descendente

        most_active_stock_1 = sorted_indices[0]
        most_active_stock_2 = sorted_indices[1]

        money_moved_1 = money_moved_per_asset[most_active_stock_1]
        money_moved_2 = money_moved_per_asset[most_active_stock_2]

        return most_active_stock_1, most_active_stock_2, money_moved_1, money_moved_2

    def crossover(self, parent1, parent2):
        """
        Realiza un crossover centrado en las mayores compras y ventas.

        Parameters:
        - parent1, parent2: np.ndarray, matrices n_assets x n_days (cromosomas padres).
        - crossover_prob: float, probabilidad de que ocurra el crossover.

        Returns:
        - child1, child2: np.ndarray, cromosomas hijos resultantes.
        """
        
        try : 
            n_assets, n_days = parent1.shape
            keep_on = True
            iteration = 0

            # Crear copias de los cromosomas padres para los hijos
            child1, child2 = parent1.copy(), parent2.copy()
            logging.debug(f"Parent1 shape: {parent1.shape}, Parent2 shape: {parent2.shape}")

            prob = np.random.random()
            # Identificar los dos activos más activos de cada cromosoma
            active1_c1, active2_c1, money1_c1, money2_c1 = self.get_most_active_stocks(
                parent1
            )
            active1_c2, active2_c2, money1_c2, money2_c2 = self.get_most_active_stocks(
                parent2
            )

            # Determinar cuál cromosoma tiene el activo con mayor actividad
            if money1_c1 > money1_c2:
                bigger_index, smaller_index = active1_c1, active1_c2
                c_bigger, c_smaller = child1, child2
                # print("Bigger is c1")
            else:
                bigger_index, smaller_index = active1_c2, active1_c1
                c_bigger, c_smaller = child2, child1
                
            if prob >= self.crossover_prob:
                logging.info("Crossover skipped due to probability threshold.")
                return [child1, child2]  # Retornar sin cambios
            both_valid = False
            while iteration < 10 or (both_valid) :
                    # print("Bigger is c2")

                ### Bigger chromosome modification ####

                # day_of_max_purchase = np.argmax(c_bigger[bigger_index] > 0) # Encontrar el día(columna) con la mayor compra para el activo más activo
                # Filtrar solo los valores mayores que 0
                try :
                    positive_indices = np.where(c_bigger[bigger_index] > 0)[
                        0
                    ]  # Índices donde hay valores positivos
                    if positive_indices.size == 0:
                        return [child1, child2]
                        raise ValueError(
                            f"No hay valores positivos en el activo el cromosoma grande {bigger_index}"
                        )
                        

                    # Encontrar el índice del valor máximo positivo
                    day_of_max_purchase = positive_indices[
                        np.argmax(c_bigger[bigger_index][positive_indices])
                    ]
                    max_value = c_bigger[bigger_index][day_of_max_purchase]

                    # Imprimir los resultados
                    # print(f"Day of max purchase: {day_of_max_purchase}, bigger_index: {bigger_index}, max_value: {max_value}")

                    big_buy_price = self.january.iloc[day_of_max_purchase, bigger_index]
                    units_bought = c_bigger[bigger_index, day_of_max_purchase]
                    usd_spent = (
                        big_buy_price * units_bought
                    )  # Calcular cuánto se gastó en esa compra

                    # Ver cuántas unidades se podrían comprar del otro activo más activo
                    price_next_day = self.january.iloc[day_of_max_purchase + 1, smaller_index]
                    could_buy_next_day = usd_spent // price_next_day
                    purchase_amount = max(
                        1, int(could_buy_next_day // 2)
                    )  # Comprar la mitad como medida de equilibrio

                    # Ajustar el cromosoma más activo
                    c_bigger[
                        bigger_index, day_of_max_purchase
                    ] -= purchase_amount  # Luego hay que buscar la próxima venta en valor absoluto > compra, y le sumamos el valor de compra para compensar

                    for day_index in range(day_of_max_purchase + 1, n_days - 1):
                        # print(f"c_bigger[bigger_index][day_index] = {c_bigger[bigger_index][day_index]},purchase_amount = {purchase_amount},day_index = {day_index}")
                        if (
                            abs(c_bigger[bigger_index][day_index]) >= purchase_amount
                            and c_bigger[bigger_index][day_index] < 0
                        ):

                            c_bigger[bigger_index][day_index] += purchase_amount

                    c_bigger[smaller_index][
                        day_of_max_purchase + 1
                    ] = purchase_amount  # We buy the other biggest stock
                    sell_day = np.random.randint(day_of_max_purchase + 1, n_days - 1)
                    c_bigger[smaller_index][
                        sell_day
                    ] -= purchase_amount  # We sell the other biggest stock

                    ### Smaller chromosome modification ####
                    # Ajustar el cromosoma menos activo
                    positive_indices = np.where(c_smaller[smaller_index] > 0)[
                        0
                    ]  # Índices donde hay valores positivos
                    if positive_indices.size == 0:
                        return [child1, child2]
                        raise ValueError(
                            f"No hay valores positivos en el activo del cromosoma pequeño {smaller_index}"
                        )
                        
                    # Encontrar el índice del valor máximo positivo
                    day_of_max_purchase_smaller = positive_indices[
                        np.argmax(c_smaller[smaller_index][positive_indices])
                    ]
                    max_value = c_smaller[smaller_index][day_of_max_purchase_smaller]
                    # print(f"Day of max purchase smaller : {day_of_max_purchase_smaller},smaller_index : {smaller_index}")

                    small_buy_price = self.january.iloc[
                        day_of_max_purchase_smaller, smaller_index
                    ]
                    small_units_bought = c_smaller[smaller_index, day_of_max_purchase_smaller]
                    small_usd_spent = small_buy_price * small_units_bought

                    # Comprar el otro activo
                    other_price_next_day = self.january.iloc[
                        day_of_max_purchase_smaller + 1, bigger_index
                    ]
                    other_could_buy_next_day = small_usd_spent // other_price_next_day
                    other_purchase_amount = max(1, int(other_could_buy_next_day // 2))

                    c_smaller[smaller_index][
                        day_of_max_purchase_smaller
                    ] -= other_purchase_amount  # Luego hay que buscar la próxima venta en valor absoluto > compra, y le sumamos el valor de compra para compensar
                    for day_index in range(day_of_max_purchase_smaller + 1, n_days - 1):

                        if (
                            abs(c_smaller[smaller_index][day_index]) >= other_purchase_amount
                            and c_smaller[smaller_index][day_index] < 0
                        ):
                            c_smaller[smaller_index][day_index] += other_purchase_amount

                    c_smaller[bigger_index][
                        day_of_max_purchase_smaller + 1
                    ] = other_purchase_amount
                    sell_day = np.random.randint(day_of_max_purchase_smaller + 1, n_days - 1)
                    c_smaller[bigger_index][sell_day] -= other_purchase_amount
                    
                    valid_bigger = self.check_chromosome(c_bigger) 
                    valid_smaller = self.check_chromosome(c_smaller)
                    both_valid = valid_bigger and valid_smaller
                    iteration += 1
                    
            
                    
                except Exception as e:
                    logging.error(f"Error during crossover iteration {iteration}: {str(e)}")
                    return [child1, child2]  # Retornar los originales en caso de error
            return [c_bigger,c_smaller]

        except Exception as e :
            logging.error(f"Unexpected error in crossover: {str(e)}")
            
            return [parent1, parent2]  # Asegura siempre un retorno válido


    def best_average_worst(self, fitness):
        best_chromosome = 0
        best_fitness = fitness[0]
        worst_fitness = fitness[0]
        average_fitness = sum(fitness) / self.population_size
        for i in range(len(fitness)):
            if self.is_better_than(fitness[i], best_fitness):
                best_chromosome = i
                best_fitness = fitness[i]
            if self.is_better_than(worst_fitness, fitness[i]):
                worst_fitness = fitness[i]
        return best_chromosome, best_fitness, average_fitness, worst_fitness

    def genetic_algorithm(self, gamma):
        generation = 0
        evolution = []
        population = [
            self.generate_chromosome_randomized_with_restrictions()
            for _ in range(self.population_size)  # Ensured valid chromosome.
        ]
        fitness = [self.CARA(c, gamma) for c in population]
        best_chromosome, best_fitness, average_fitness, worst_fitness = (
            self.best_average_worst(fitness)
        )
        print(
            f">Generation {generation}: Worst: {worst_fitness:.3f} Average: {average_fitness:.3f} Best: {best_fitness:.3f}"
        )
        evolution.append([best_fitness, average_fitness, worst_fitness])
        start_time = time.time()

        while generation < self.max_gen and time.time() - start_time < self.max_time:
            generation += 1

            selected = [
                self.selection(population, fitness) for _ in range(self.population_size)
            ]
            offspring = []
            if self.elitism:
                offspring.append(population[best_chromosome])
            for i in range(0, self.population_size, 2):
                p1, p2 = selected[i], selected[(i + 1) % len(selected)]
                #print(f"P1 : {p1}",p2)
                cross_over = self.crossover(p1, p2)
                for c in cross_over:
                    self.mutate_selected_points(c)
                    if (len(offspring) < self.population_size) and (
                        self.check_chromosome(c) == True
                    ):
                        offspring.append(c)
            if len(offspring) == 0:
                logging.warning(
                    "No valid offspring generated. Replacing with random chromosomes."
                )
                offspring = [
                    self.generate_chromosome_randomized_with_restrictions()
                    for _ in range(self.population_size)
                ]
            population = offspring if len(offspring) != 0 else population
            #print(f"Population b4 CARA : {len(population)}")
            fitness = [self.CARA(c, gamma) for c in population]
            best_chromosome, best_fitness, average_fitness, worst_fitness = (
                self.best_average_worst(fitness)
            )
            print(
                f">Generation {generation}: Worst: {worst_fitness:.3f} Average: {average_fitness:.3f} Best: {best_fitness:.3f}"
            )
            evolution.append([best_fitness, average_fitness, worst_fitness])

        return population[best_chromosome], fitness[best_chromosome], evolution
    
    def calculate_money_with_fee(self,chromosome):
        """
        Calculates the money invested or earned based on a chromosome (which indicates asset purchases/sales)
        and a price matrix (which indicates the asset prices per day), considering a transaction fee.

        Parameters:
        - chromosome: np.ndarray, a matrix of size (n_assets, n_days), where each value
        indicates the number of units bought/sold of an asset on a given day.

        Returns:
        - total_money: float, the total money invested or earned, accounting for transaction fees.
        - zero_count: int, the number of days where no assets were bought or sold.
        - positive_count: int, the number of days where assets were bought.
        - negative_count: int, the number of days where assets were sold.
        - total_operations: int, the total number of buy/sell operations performed.
        """
        
        if isinstance(self.january, pd.DataFrame):
                price_matrix = self.january.values

        # Contabilizamos los valores 0, positivos y negativos
        zero_count = np.sum(chromosome == 0)
        positive_count = np.sum(chromosome > 0)
        negative_count = np.sum(chromosome < 0)

        # Calculamos el dinero total: (cantidad * precio) y restamos las tarifas de operación
        total_money = np.sum(chromosome * price_matrix.T)  
        total_money -= np.sum(chromosome != 0) * self.operational_cost  #
        total_operations = np.count_nonzero(chromosome)

        return total_money, zero_count, positive_count, negative_count,total_operations

    def execute(self, gamma):
        best_chromosome, best_fitness, evolution = self.genetic_algorithm(gamma)
        print(
            f"Execution completed! Best solution: f({best_chromosome}) = {best_fitness:.3f}"
        )

        
        resulting_usd = self.calculate_money_with_fee(best_chromosome)
        print(
             f"Initial Capital:{self.initial_capital:.2f}, Resulting USD: {resulting_usd[0]:.2f}, Days holded : {resulting_usd[1]:.2f}, Stocks bought: {resulting_usd[2]:.2f}, Stocks sold: {resulting_usd[3]:.2f}, Total operations: {resulting_usd[4]:.2f}"
         )

        plt.plot(range(len(evolution)), [x[0] for x in evolution], label="Best")
        plt.plot(range(len(evolution)), [x[1] for x in evolution], label="Average")
        plt.plot(range(len(evolution)), [x[2] for x in evolution], label="Worst")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Evolution")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    GA = GeneticAlgorithm(file_name="predicciones_deescaladas.csv", population_size=300,mutation_prob=0.5,max_gen=100,max_time=600,num_mutations=1,elitism=True,tournament_size=12,crossover_prob=0.5)
    GA.execute(gamma=0.2)
